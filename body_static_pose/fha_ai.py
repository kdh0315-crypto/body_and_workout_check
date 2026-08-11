from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================================================
# 모델 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

ENGINE_PATH = (
    BASE_DIR
    / "runtime_models"
    / "fha_mobilenet.engine"
)


# =========================================================
# AI threshold
# =========================================================

AI_T1 = 0.55
AI_T2 = 0.75


def classify_fha_ai(score):
    if score < AI_T1:
        return "AI_NORMAL"

    if score < AI_T2:
        return "AI_BORDERLINE"

    return "AI_ABNORMAL"


# =========================================================
# FHA TensorRT Classifier
# =========================================================

class FHAClassifier:

    def __init__(self, engine_path=ENGINE_PATH):

        self.engine_path = Path(engine_path)

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"FHA TensorRT engine을 찾을 수 없습니다: "
                f"{self.engine_path}"
            )

        # -------------------------------------------------
        # TensorRT engine load
        # -------------------------------------------------

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        with open(self.engine_path, "rb") as f:

            self.runtime = trt.Runtime(
                self.logger
            )

            self.engine = (
                self.runtime.deserialize_cuda_engine(
                    f.read()
                )
            )

        if self.engine is None:
            raise RuntimeError(
                "FHA TensorRT engine 로드 실패"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "TensorRT execution context 생성 실패"
            )


        # -------------------------------------------------
        # Input / Output tensor 찾기
        # -------------------------------------------------

        self.input_name = None
        self.output_name = None

        for i in range(
            self.engine.num_io_tensors
        ):

            name = (
                self.engine.get_tensor_name(i)
            )

            mode = (
                self.engine.get_tensor_mode(name)
            )

            if (
                mode
                == trt.TensorIOMode.INPUT
            ):
                self.input_name = name

            elif (
                mode
                == trt.TensorIOMode.OUTPUT
            ):
                self.output_name = name


        if self.input_name is None:
            raise RuntimeError(
                "TensorRT input tensor를 찾지 못했습니다."
            )

        if self.output_name is None:
            raise RuntimeError(
                "TensorRT output tensor를 찾지 못했습니다."
            )


        # -------------------------------------------------
        # Input shape
        # -------------------------------------------------

        self.context.set_input_shape(
            self.input_name,
            (1, 224, 224, 3),
        )


        # -------------------------------------------------
        # Output 정보
        # -------------------------------------------------

        self.output_shape = tuple(
            self.context.get_tensor_shape(
                self.output_name
            )
        )

        self.output_dtype = trt.nptype(
            self.engine.get_tensor_dtype(
                self.output_name
            )
        )

        self.output_data = np.empty(
            self.output_shape,
            dtype=self.output_dtype,
        )


        # -------------------------------------------------
        # GPU memory
        # -------------------------------------------------

        input_bytes = (
            1
            * 224
            * 224
            * 3
            * np.dtype(
                np.float32
            ).itemsize
        )

        self.d_input = cuda.mem_alloc(
            input_bytes
        )

        self.d_output = cuda.mem_alloc(
            self.output_data.nbytes
        )


        # -------------------------------------------------
        # CUDA stream
        # -------------------------------------------------

        self.stream = cuda.Stream()


        # -------------------------------------------------
        # TensorRT tensor ↔ GPU memory 연결
        # -------------------------------------------------

        self.context.set_tensor_address(
            self.input_name,
            int(self.d_input),
        )

        self.context.set_tensor_address(
            self.output_name,
            int(self.d_output),
        )


        print(
            "[FHA AI] TensorRT engine loaded"
        )

        print(
            f"[FHA AI] engine: "
            f"{self.engine_path}"
        )

        print(
            f"[FHA AI] input: "
            f"{self.input_name}"
        )

        print(
            f"[FHA AI] output: "
            f"{self.output_name}"
        )


    # =====================================================
    # Preprocess
    # =====================================================

    @staticmethod
    def preprocess(frame):

        if frame is None:
            raise ValueError(
                "FHA AI input frame is None"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # MobileNetV2 입력 크기
        image = cv2.resize(
            image,
            (224, 224),
        )

        image = image.astype(
            np.float32
        )

        # MobileNetV2 preprocess_input
        # 0~255 -> -1~1
        image = (
            image / 127.5
            - 1.0
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        return np.ascontiguousarray(
            image
        )


    # =====================================================
    # TensorRT inference
    # =====================================================

    def predict(self, frame):

        input_data = self.preprocess(
            frame
        )

        cuda.memcpy_htod_async(
            self.d_input,
            input_data,
            self.stream,
        )

        success = (
            self.context.execute_async_v3(
                stream_handle=
                self.stream.handle
            )
        )

        if not success:
            raise RuntimeError(
                "FHA TensorRT inference 실패"
            )

        cuda.memcpy_dtoh_async(
            self.output_data,
            self.d_output,
            self.stream,
        )

        self.stream.synchronize()

        score = float(
            self.output_data.ravel()[0]
        )

        return score


    # =====================================================
    # score + state
    # =====================================================

    def predict_result(self, frame):

        score = self.predict(
            frame
        )

        ai_result = classify_fha_ai(
            score
        )

        return {
            "score": score,
            "result": ai_result,
        }


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    classifier = FHAClassifier()

    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("카메라 프레임을 읽지 못했습니다.")

    result = classifier.predict_result(frame)

    print(result)

    cap.release()
