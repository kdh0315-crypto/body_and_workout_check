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
from module.workout_checker import get_exercise_checker, ActionRecognizer
from module.workout_sel import workout_sel


# ---- LLM 처방을 흉내 낸 테스트용 운동 목록 ----
# 실제로는 prescribe_workouts() 결과가 이 형식으로 들어온다.
# 테스트가 빨리 끝나도록 count/sets/rest 를 작게 잡았다.
MOCK_WORKOUT = {
    "recommendations": [
        {"exercise": "pushup", "priority": 1, "count": 5, "unit": "reps", "sets": 2, "reason": "테스트"},
        {"exercise": "pushup", "priority": 2, "count": 10, "unit": "reps", "sets": 1, "reason": "테스트2"},
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
            recognized_action = None
            if has_landmarks(results):
                landmarks = results.pose_landmarks[0]

                # LSTM으로 실제 동작 인식 (버퍼가 덜 찼으면 None)
                action_result = recognizer.update(landmarks)
                if action_result is not None:
                    recognized_action, _confidence = action_result

                # selector가 지시한 운동(checker.name)과 인식된 운동이 같을 때만 카운트
                if recognized_action == checker.name:
                    checker.update(landmarks, w, h)

                frame = draw_skeleton(frame, results)

            session = checker.session
            cv2.putText(frame, f"Exercise: {checker.name}  (Recognized: {recognized_action or '...'})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Set: {session.rep_count}/{session.target_reps}  "
                               f"Count: {session.count}/{session.target_count}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if session.resting:
                cv2.putText(frame, f"REST: {session.rest_remaining():.1f}s",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

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