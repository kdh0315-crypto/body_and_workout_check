import json
import ollama
import cv2

import time

from ollama_op import *
from workout_sel import *
from mediapipe_op import *
from test_fn import *

def capture_image(frame):
    # copy the frame data
    captured = frame.copy()
    return captured

if __name__ == "__main__":
    # --- 임의의 측정값 (실제로는 MediaPipe 계산 결과로 대체) ---
    # 아래는 일부러 몇 개를 이상값으로 넣은 예시.
    sample_metrics = {
        "shoulder_tilt":     5.2,   # 정상 0~2 초과 -> 이상(어깨 비대칭)
        "knee_valgus":       2.0,   # 정상
        "forward_head":      42.0,  # 정상 55~65 미달 -> 이상(거북목 심함)
        "round_shoulder":    11.0,   # 정상 0~10 초과 -> 이상(라운드숄더)
        "thoracic_kyphosis": 38.0,  # 정상
        "pelvic_tilt_ant":   7.0,   # 정상
    }

    # 

    # capture image to test body form
    cap_form = CaptureForm()
    front_img = None
    side_img  = None

    # get camera
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Unable to open Camera. Check the device number"

    while cap.isOpened():
        ret, frame = cap.read()
        assert ret, "Unable to read frame"

        # mirror mode to image - to see customer's pose easily
        frame = cv2.flip(frame, 1)
        # get image size
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # make frame to read only mode for performance
        # when use mediapipe to draw joint image, need to make it True
        rgb_frame.flags.writeable = False

        # Show image
        cv2.imshow('Exercise Pose Test', frame)

        # Get keyboard input to control camera operation
        key = cv2.waitKey(1) & 0xFF

        # Take image & make joint image Key
        result = cap_form.capture_form(key, frame)

        # Split image & save to see image
        if result is not None:
            front_img, side_img = result
            print("정면/측면 촬영 완료 — 저장 중")

            # 원본 저장
            cv2.imwrite("front_raw.jpg", front_img)
            cv2.imwrite("side_raw.jpg", side_img)

        # Escape Key
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()