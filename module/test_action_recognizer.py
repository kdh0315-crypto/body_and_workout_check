"""
ActionRecognizer(LSTM 운동 종류 인식) + Checker(rep 카운트)를 확인하는 스크립트.

workout_sel(운동 처방/순서/휴식 FSM)은 아직 빼고, 카메라 -> MediaPipe ->
ActionRecognizer -> (인식된 운동에 해당하는) Checker 로만 이어지는 구성이다.
squat/pushup/lunge Checker를 전부 미리 만들어두고, 매 프레임 ActionRecognizer가
인식한 운동의 Checker에만 프레임을 흘려보내 카운트한다 — 즉 어떤 운동을 선택할지는
순전히 인식 결과가 정하고, workout_sel처럼 미리 정해진 순서/목표 세트는 없다.

조작:
    'r' - 시퀀스 버퍼 + 모든 Checker 리셋
    'q' - 종료
"""

import time

import cv2

from module.mediapipe_op import extract_landmarks_video, has_landmarks, draw_skeleton
from module.workout_checker import ActionRecognizer, get_exercise_checker

# workout_sel 없이 그냥 세 종목 Checker를 다 만들어두고 인식된 걸로만 카운트한다.
# target_reps/count를 크게 잡아서 테스트 중에 session.done으로 멈추지 않게 한다.
CHECKED_EXERCISES = ["squat", "pushup", "lunge"]

# 10프레임으로 새로 학습한 엔진을 이 스크립트에서만 먼저 검증한다.
# workout_checker.py의 공용 기본값(LSTM_ENGINE_PATH/SEQ_LEN=15프레임)은 그대로 두고,
# 여기서만 ActionRecognizer 생성 시 오버라이드 — test_sel_checker.py는 영향 없음.
LSTM_ENGINE_PATH = "models/lstm_10frame.trt"
SEQ_LEN = 10


HUD_W = 430   # 좌측 HUD 패널 폭 (배경 박스 + 줄바꿈 기준)


def draw_hud_bg(frame, x0, y0, w, h, alpha=0.6):
    """해당 영역을 반투명 검정으로 깔아서 그 위 텍스트가 어떤 배경에서도 잘 보이게 한다."""
    x1, y1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    roi = frame[y0:y1, x0:x1]
    black = (roi * 0).astype(roi.dtype)
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, black, alpha, 0)


def put_text(frame, text, pos, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_probs(frame, actions, probs, top_idx):
    """클래스별 확률을 막대로 표시."""
    x0, y0, bar_w, bar_h = 10, 55, 180, 26
    for i, (name, p) in enumerate(zip(actions, probs)):
        y = y0 + i * (bar_h + 8)
        color = (0, 220, 0) if i == top_idx else (90, 90, 90)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y), (x0 + int(bar_w * p), y + bar_h), color, -1)
        put_text(frame, f"{name}: {p * 100:4.1f}%", (x0 + bar_w + 10, y + bar_h - 6),
                  scale=0.6, color=(255, 255, 255), thickness=2)


def draw_check_result(frame, active_name, check_result, y0):
    """활성 Checker가 반환한 상태/각도를 표시 (카운트가 왜 안 되는지 디버깅용).
    check_result가 None이면 이유(인식 안 됨 / 랜드마크 가시성 부족)를 대신 표시한다."""
    x0 = 10
    if active_name is None:
        put_text(frame, "checker: no exercise recognized", (x0, y0), scale=0.6, color=(80, 80, 255))
        return
    if check_result is None:
        put_text(frame, f"checker[{active_name}]: visibility too low", (x0, y0), scale=0.6, color=(80, 80, 255))
        return
    if "angles" not in check_result:
        put_text(frame, f"checker[{active_name}]: resting/done", (x0, y0), scale=0.6, color=(0, 165, 255))
        return

    state = check_result.get("state")
    if state is not None:
        put_text(frame, f"state: {state}", (x0, y0), scale=0.65, color=(0, 255, 255))
        y0 += 32
    for name, angle in check_result["angles"].items():
        put_text(frame, f"{name}: {angle:.0f} deg", (x0, y0), scale=0.6, color=(0, 220, 255))
        y0 += 28


def draw_checker_status(frame, checkers, active_name, y0):
    """운동별 세션 상태(카운트)를 표시. 현재 인식된 운동만 강조."""
    x0 = 10
    for i, name in enumerate(CHECKED_EXERCISES):
        session = checkers[name].session
        color = (0, 255, 0) if name == active_name else (170, 170, 170)
        put_text(
            frame,
            f"{name}: Set {session.rep_count}/{session.target_reps}  "
            f"Count {session.count}/{session.target_count}",
            (x0, y0 + i * 30), scale=0.6, color=color,
        )


def main():
    recognizer = ActionRecognizer(engine_path=LSTM_ENGINE_PATH, seq_len=SEQ_LEN)
    checkers = {
        name: get_exercise_checker(name, target_reps=99, target_count=99, rest_seconds=0)
        for name in CHECKED_EXERCISES
    }

    # ----- 카메라 캡처 속도 최적화 -----
    # 기본 백엔드/포맷으로 열면 드라이버가 저FPS 포맷(YUYV 등)을 골라 cap.read() 자체가
    # 느릴 수 있다. MJPG + 해상도를 명시하면 대부분의 USB 웹캠에서 훨씬 빨라진다.
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 오래된 프레임이 쌓이지 않도록
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print(f"Capture: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps (요청값, 실제 카메라가 지원 안 하면 다를 수 있음)")

    start_time = time.perf_counter()
    prev_ts = -1

    # 구간별 소요시간 EMA (어디가 병목인지 화면에서 바로 확인용)
    t_capture = t_pose = t_lstm = t_total = 0.0
    TIMING_ALPHA = 0.1

    print("동작 인식 + Checker 카운트 테스트 (workout_sel 없음). 'r' 리셋 / 'q' 종료.")

    while cap.isOpened():
        loop_start = time.perf_counter()

        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_capture_now = time.perf_counter() - t0
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        ts = int((time.perf_counter() - start_time) * 1000)
        if ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts

        t0 = time.perf_counter()
        results = extract_landmarks_video(frame, ts)
        t_pose_now = time.perf_counter() - t0
        landmarks = results.pose_landmarks[0] if has_landmarks(results) else None

        t0 = time.perf_counter()
        result = recognizer.update(landmarks)
        t_lstm_now = time.perf_counter() - t0
        frame = draw_skeleton(frame, results)

        t_capture = TIMING_ALPHA * t_capture_now + (1 - TIMING_ALPHA) * t_capture
        t_pose = TIMING_ALPHA * t_pose_now + (1 - TIMING_ALPHA) * t_pose
        t_lstm = TIMING_ALPHA * t_lstm_now + (1 - TIMING_ALPHA) * t_lstm
        t_total = TIMING_ALPHA * (time.perf_counter() - loop_start) + (1 - TIMING_ALPHA) * t_total

        # ----- HUD 배경(반투명 검정) 먼저 깔기: 어떤 배경에서도 글씨가 잘 보이게 -----
        draw_hud_bg(frame, 0, 0, HUD_W, 415)
        draw_hud_bg(frame, 0, frame.shape[0] - 55, HUD_W, 55)

        active_name = None
        check_result = None
        if result is None:
            put_text(frame, f"Buffering... {len(recognizer.sequence)}/{recognizer.seq_len}",
                      (10, 32), scale=0.85, color=(0, 200, 255))
        else:
            action_name, confidence = result
            top_idx = int(recognizer.ema_probs.argmax())
            put_text(frame, f"Recognized: {action_name} ({confidence * 100:.1f}%)",
                      (10, 32), scale=0.85, color=(0, 255, 0))
            draw_probs(frame, recognizer.actions, recognizer.ema_probs, top_idx)

            # 인식된 운동이 Checker가 있는 종목이면, 그 Checker에만 프레임을 흘려서
            # 레퍼런스(FormFit)처럼 매 프레임 각도 기반으로 카운트한다.
            if action_name in checkers and landmarks is not None:
                active_name = action_name
                check_result = checkers[action_name].update(landmarks, w, h)

        draw_checker_status(frame, checkers, active_name, y0=195)
        draw_check_result(frame, active_name, check_result, y0=280)

        fps = 1.0 / t_total if t_total > 0 else 0.0
        put_text(frame,
                 f"cap {t_capture*1000:.0f}ms | pose {t_pose*1000:.0f}ms | "
                 f"lstm {t_lstm*1000:.0f}ms | total {t_total*1000:.0f}ms ({fps:.1f} FPS)",
                 (10, frame.shape[0] - 35), scale=0.55, color=(0, 255, 255))
        put_text(frame, "Press 'r' reset / 'q' quit",
                  (10, frame.shape[0] - 12), scale=0.55, color=(255, 255, 255))
        cv2.imshow("Action Recognizer Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            recognizer.reset()
            for checker in checkers.values():
                checker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
