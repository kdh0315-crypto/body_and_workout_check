from module.workout_sel import *


def _mock_response():
    """Ollama 출력을 흉내 낸 가짜 응답 (서버 없이 테스트용)."""
    return {
        "recommendations": [
            {"exercise": "승모 스트레칭", "priority": 1, "reason": "거북목 개선"},
            {"exercise": "W/Y레이즈",    "priority": 2, "reason": "라운드숄더 교정"},
            {"exercise": "플랭크",        "priority": 3, "reason": "코어 안정화"},
        ]
    }


def test_workout_sel():
    print("=" * 40)
    print("workout_sel 상태 전이 테스트")
    print("=" * 40)

    sel = workout_sel()
    assert sel.state == "idle", "초기 상태는 idle이어야 함"
    assert sel.current_workout() is None, "로드 전엔 현재 운동 없음"
    print(f"[초기] state={sel.state}, current={sel.current_workout()}")

    # --- 로드 ---
    resp = _mock_response()
    sel.load_workout({"recommendations": resp["recommendations"]})

    assert sel.state == "work", "로드 후 work 상태여야 함"
    print(f"\n[로드 후] state={sel.state}, 총 {len(sel.workouts)}개")

    # priority 순 정렬 확인
    priorities = [w["priority"] for w in sel.workouts]
    assert priorities == sorted(priorities), "priority 오름차순 정렬돼야 함"
    print(f"정렬된 우선순위: {priorities}")

    # --- 순차 배출 ---
    print("\n[운동 진행]")
    expected = ["승모 스트레칭", "W/Y레이즈", "플랭크"]
    for i, name in enumerate(expected):
        cur = sel.current_workout()
        assert cur is not None, f"{i}번째 운동이 있어야 함"
        assert cur["exercise"] == name, f"순서 불일치: {cur['exercise']} != {name}"
        print(f"  {i+1}번째: {cur['exercise']} (state={sel.state})")

        nxt = sel.next_workout(work_done=True)
        if i < len(expected) - 1:
            assert nxt is not None, "아직 다음 운동이 남아야 함"
        else:
            assert nxt is None, "마지막 운동 후엔 None"

    # --- 완료 후 상태 ---
    assert sel.state == "idle", "모두 끝나면 idle로 복귀"
    assert sel.current_workout() is None, "완료 후 현재 운동 없음"
    print(f"\n[완료] state={sel.state}, current={sel.current_workout()}")

    print("\n✅ 상태 전이 테스트 통과")


def test_edge_cases():
    print("\n" + "=" * 40)
    print("엣지 케이스 테스트")
    print("=" * 40)

    # 1) 빈 추천 리스트
    sel = workout_sel()
    sel.load_workout({"recommendations": []})
    assert sel.state == "idle", "빈 리스트면 idle 유지"
    assert sel.current_workout() is None
    print("[빈 추천] state=idle, current=None ✅")

    # 2) work_done=False면 전진하지 않음
    sel = workout_sel()
    sel.load_workout({"recommendations": [
        {"exercise": "스쿼트", "priority": 1, "reason": "테스트"},
        {"exercise": "플랭크", "priority": 2, "reason": "테스트"},
    ]})
    first = sel.current_workout()
    same = sel.next_workout(work_done=False)   # 완료 안 됨
    assert same["exercise"] == first["exercise"], "미완료 시 같은 운동 유지"
    print(f"[미완료] '{same['exercise']}' 유지 ✅")

    # 3) idle 상태에서 next_workout 호출해도 안전
    sel = workout_sel()
    assert sel.next_workout(work_done=True) is None, "idle에서 호출해도 None"
    print("[idle 호출] None 반환 ✅")

    print("\n✅ 엣지 케이스 테스트 통과")