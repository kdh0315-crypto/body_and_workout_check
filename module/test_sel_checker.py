"""
workout_sel + workout_checker 연동 테스트 (GUI 없이 검증용)

흐름:
  1) workout_sel 에 운동 목록(LLM 처방 형식)을 로드
  2) 현재 운동으로 체커를 생성
  3) 카메라로 실시간 판별 → ActionRecognizer(LSTM)가 인식한 운동이
     selector가 지시한 운동(checker.name)과 같을 때만 매 프레임 checker.update()로
     카운트 (레퍼런스인 FormFit과 동일한 각도 기반 방식)
  4) 한 운동의 모든 세트 완료(session.done) 시 workout_sel.next_workout() 으로
     다음 운동을 받아 체커 교체
  5) 목록이 끝나면 종료

주의:
  운동 이름은 workout_checker.get_exercise_checker / ActionRecognizer.ACTIONS와
  동일하게 "squat" / "pushup" / "lunge" 를 사용해야 한다 (EXERCISE_POOL도 동일).
  바이셉컬·플랭크는 종목 구성에서 제외되어 더 이상 지원하지 않는다.

조작:
  'r' - 현재 운동 리셋
  'q' - 종료
"""

import time
import cv2

from module.mediapipe_op import (
    extract_landmarks_video,
    has_landmarks,
    draw_skeleton,
)
from module.workout_checker import get_exercise_checker, ActionRecognizer, update_with_recognition
from module.workout_sel import workout_sel


HUD_W = 430   # 좌측 HUD 패널 폭


def draw_hud_bg(frame, x0, y0, w, h, alpha=0.6):
    """해당 영역을 반투명 검정으로 깔아서 그 위 텍스트가 어떤 배경에서도 잘 보이게 한다."""
    x1, y1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    roi = frame[y0:y1, x0:x1]
    black = (roi * 0).astype(roi.dtype)
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, black, alpha, 0)


def put_text(frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_check_result(frame, check_result, y0):
    """checker.update()가 돌려준 상태/각도를 표시 (레퍼런스 각도값이 실제로 어떻게
    변하고 있는지, 카운트가 왜 되거나 안 되는지 바로 확인하기 위한 디버깅용)."""
    x0 = 10
    if check_result is None:
        put_text(frame, "checker: 인식된 운동과 처방 운동이 달라 카운트 안 함", (x0, y0),
                  scale=0.55, color=(80, 80, 255))
        return y0 + 26
    if "angles" not in check_result:
        put_text(frame, "checker: resting/done", (x0, y0), scale=0.55, color=(0, 165, 255))
        return y0 + 26

    state = check_result.get("state")
    if state is not None:
        put_text(frame, f"state: {state}", (x0, y0), scale=0.6, color=(0, 255, 255))
        y0 += 26
    for name, angle in check_result["angles"].items():
        put_text(frame, f"{name}: {angle:.0f} deg", (x0, y0), scale=0.55, color=(0, 220, 255))
        y0 += 22
    return y0 + 4


def draw_feedback(frame, checker, y0):
    """세션 누적 피드백(session.error_counter)을 표시.
    'Good form' 카운트와 각 오류 메시지별 발생 횟수를 그대로 보여준다 —
    운동별 피드백 로직이 실제로 맞는 메시지를 내는지 여기서 눈으로 확인한다."""
    x0 = 10
    put_text(frame, "Feedback (누적):", (x0, y0), scale=0.6, color=(255, 255, 0))
    y0 += 26
    error_counter = checker.session.error_counter
    if not error_counter:
        put_text(frame, "  (아직 기록 없음)", (x0, y0), scale=0.55, color=(180, 180, 180))
        return y0 + 24
    for err, cnt in error_counter.items():
        color = (0, 255, 0) if err == "Good form" else (0, 140, 255)
        put_text(frame, f"  {err}: {cnt}", (x0, y0), scale=0.55, color=color)
        y0 += 22
    return y0 + 4


# ---- LLM 처방을 흉내 낸 테스트용 운동 목록 ----
# 실제로는 prescribe_workouts() 결과가 이 형식으로 들어온다.
# 테스트가 빨리 끝나도록 count/sets/rest 를 작게 잡았다.
MOCK_WORKOUT = {
    "recommendations": [
        {"exercise": "lunge", "priority": 1, "count": 5, "unit": "reps", "sets": 2, "reason": "테스트"},
        {"exercise": "lunge", "priority": 2, "count": 10, "unit": "reps", "sets": 1, "reason": "테스트2"},
    ]
}

REST_SECONDS = 5   # 테스트용 짧은 쉬는 시간


def make_checker(workout_item):
    """workout_sel이 준 운동 항목으로 체커를 생성."""
    return get_exercise_checker(
        workout_item["exercise"],
        target_reps=workout_item.get("sets", 1),
        target_count=workout_item.get("count", 10),
        rest_seconds=REST_SECONDS,
    )


def main():
    # 1) workout_sel 에 목록 로드
    selector = workout_sel()
    selector.load_workout(MOCK_WORKOUT)

    current = selector.current_workout()
    if current is None:
        print("추천된 운동이 없습니다.")
        return

    checker = make_checker(current)
    print(f"\n===== 현재 운동: {current['exercise']} "
          f"({current['count']}회 x {current['sets']}세트) =====\n")

    # LSTM으로 "지금 실제로 하고 있는 운동"을 인식 → selector가 지시한 운동과
    # 일치할 때만 checker로 카운트한다.
    recognizer = ActionRecognizer()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    start_time = time.perf_counter()
    prev_ts = -1
    all_finished = False

    # 운동 간 휴식 상태
    between_rest = False
    between_rest_start = None
    BETWEEN_EXERCISE_REST = 10   # 운동 사이 쉬는 시간(초)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        ts = int((time.perf_counter() - start_time) * 1000)
        if ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts

        if all_finished:
            cv2.putText(frame, "ALL WORKOUTS DONE",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        elif between_rest:
            # ----- 운동 간 휴식 중 -----
            remaining = BETWEEN_EXERCISE_REST - (time.time() - between_rest_start)
            if remaining <= 0:
                between_rest = False           # 휴식 끝 → 다음 운동 시작
                print(f"\n===== 다음 운동: {checker.name} 시작 =====\n")
            else:
                cv2.putText(frame, f"NEXT EXERCISE IN: {remaining:.1f}s",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        else:
            # ----- 운동 판별 중 -----
            results = extract_landmarks_video(frame, ts)
            landmarks = results.pose_landmarks[0] if has_landmarks(results) else None

            # 인식 -> (인식된 운동이 checker.name과 같을 때만) 카운트, 한 번에 처리
            recognized_action, _confidence, check_result = update_with_recognition(
                recognizer, checker, landmarks, w, h)

            if landmarks is not None:
                frame = draw_skeleton(frame, results)

            draw_hud_bg(frame, 0, 0, HUD_W, 320)

            session = checker.session
            put_text(frame, f"Exercise: {checker.name}  (Recognized: {recognized_action or '...'})",
                      (10, 26), scale=0.65, color=(255, 255, 0))
            put_text(frame, f"Set: {session.rep_count}/{session.target_reps}  "
                             f"Count: {session.count}/{session.target_count}",
                      (10, 56), scale=0.65, color=(0, 255, 255))
            if session.resting:
                put_text(frame, f"REST: {session.rest_remaining():.1f}s",
                          (10, 86), scale=0.7, color=(0, 165, 255))

            y = draw_check_result(frame, check_result, y0=112)
            draw_feedback(frame, checker, y0=y)

            # ----- 한 운동 완료 → 다음 운동 요청 + 운동 간 휴식 시작 -----
            if session.done:
                print(f"[{checker.name}] 완료. 다음 운동 요청.")
                nxt = selector.next_workout(work_done=True)
                if nxt is None:
                    all_finished = True
                    print("\n===== 모든 운동 완료! =====\n")
                else:
                    checker = make_checker(nxt)     # 다음 체커 미리 준비
                    recognizer.reset()               # 다음 운동 인식을 위해 시퀀스 버퍼 초기화
                    between_rest = True              # 운동 간 휴식 시작
                    between_rest_start = time.time()
                    print(f"\n----- 운동 간 휴식 {BETWEEN_EXERCISE_REST}s -----\n")

        cv2.putText(frame, "Press 'r' reset / 'q' quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Workout Sel + Checker Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            checker.reset()
            recognizer.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()