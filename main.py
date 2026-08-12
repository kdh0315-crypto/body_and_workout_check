"""
Jetson 실시간 운동 인식 (LSTM + TensorRT)

로컬 웹캠 → MediaPipe Tasks로 132차원 좌표 추출 → 30프레임 시퀀스 →
TRT 엔진으로 변환한 LSTM으로 운동 분류.

사전 준비 (한 번만):
    1) Keras -> ONNX:
       python -c "import tensorflow as tf, tf2onnx; \
         m=tf.keras.models.load_model('model_new.keras'); \
         tf2onnx.convert.from_keras(m, input_signature=(tf.TensorSpec((None,30,132),tf.float32,name='input'),), output_path='model_lstm.onnx')"
    2) ONNX -> TRT (반드시 이 Jetson에서 빌드):
       trtexec --onnx=model_lstm.onnx --saveEngine=lstm_classifier.trt --fp16

조작:
    'q' - 종료
"""

import cv2
import numpy as np
import time

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (CUDA 컨텍스트 초기화)


# ===================== 설정 =====================
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
LSTM_ENGINE_PATH = "models/lstm.trt"      # 이 Jetson에서 빌드한 엔진

ACTIONS = np.array(['pushup', 'squat', 'lunge', 'noactions'])
SEQ_LEN = 30
NUM_FEATURES = 132
THRESHOLD = 0.5


# ===================== TRT LSTM 추론 클래스 =====================
# 기존 workout_checker.py의 TRTModel과 동일한 구조.
# 입력이 (1, 30, 132) 시퀀스라는 점만 다르다.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTLSTMModel:
    def __init__(self, engine_path, seq_len, num_features, num_classes):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_size = seq_len * num_features    # 30 * 132
        self.output_size = num_classes              # 4

        self.d_input = cuda.mem_alloc(self.input_size * np.float32().itemsize)
        self.d_output = cuda.mem_alloc(self.output_size * np.float32().itemsize)
        self.stream = cuda.Stream()

    def predict(self, seq):
        """seq: (30, 132) float32 -> 확률 (num_classes,)"""
        input_array = np.ascontiguousarray(seq, dtype=np.float32)
        output_array = np.empty(self.output_size, dtype=np.float32)

        cuda.memcpy_htod_async(self.d_input, input_array, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_array, self.d_output, self.stream)
        self.stream.synchronize()
        return output_array


lstm_model = TRTLSTMModel(LSTM_ENGINE_PATH, SEQ_LEN, NUM_FEATURES, len(ACTIONS))
print("TRT LSTM 엔진 로드 완료")


# ===================== MediaPipe Tasks (VIDEO 모드) =====================
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def draw_landmarks_on_frame(frame, result):
    """MediaPipe가 실제로 감지한 관절을 화면에 그려서 눈으로 확인할 수 있게 함."""
    if not result.pose_landmarks:
        return
    landmark_list = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in result.pose_landmarks[0]
        ]
    )
    mp_drawing.draw_landmarks(
        frame,
        landmark_list,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )


def extract_keypoints(result):
    """Tasks 결과 -> 132차원. 감지 실패 시 0으로."""
    if not result.pose_landmarks:
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    lms = result.pose_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                    dtype=np.float32).flatten()


# ===================== 메인 루프 =====================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    sequence = []
    current_action = ""
    current_prob = 0.0

    start_time = time.perf_counter()
    prev_ts = -1

    print("실시간 인식 시작. 'q' 종료.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        ts = int((time.perf_counter() - start_time) * 1000)
        if ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts)
        draw_landmarks_on_frame(frame, result)

        keypoints = extract_keypoints(result)
        sequence.append(keypoints)
        sequence = sequence[-SEQ_LEN:]

        if len(sequence) == SEQ_LEN:
            probs = lstm_model.predict(np.array(sequence))    # (30,132) -> (4,)
            idx = int(np.argmax(probs))
            current_prob = float(probs[idx])
            current_action = ACTIONS[idx] if current_prob > THRESHOLD else "..."

        cv2.rectangle(frame, (0, 0), (320, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"{current_action}  {current_prob*100:.0f}%",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Exercise Recognition (LSTM+TRT)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()