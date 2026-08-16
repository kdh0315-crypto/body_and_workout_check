import json
import ollama

REF_RANGES = {
    "shoulder_tilt":     ("Shoulder lateral tilt",              0,  2, "abs"),
    "knee_valgus":       ("Knee alignment deviation (L/R avg)", 0,  5, "high"),
    "forward_head":      ("Forward head angle (ear-shoulder-hip)", 55, 65, "low"),
    "round_shoulder":    ("Round shoulder angle",               0, 10, "high"),
    "thoracic_kyphosis": ("Thoracic kyphosis angle",           30, 40, "high"),
    "pelvic_tilt_ant":   ("Anterior pelvic tilt angle",         5, 10, "high"),
}

# 운동 3종 + 안전 범위 (횟수는 코드에서 이 범위로 clamp)
# weight_capable: 덤벨 등 외부 중량을 들고 할 수 있는 운동인지 (푸시업은 몸으로 미는 동작이라 불가)
# weight_range: weight_capable인 운동에 한해, 권장 중량(kg)의 최소~최대 범위.
#               런지는 편측(한쪽 다리) 운동이라 균형/안전을 고려해 스쿼트보다 낮게 설정.
EXERCISE_POOL = {
    "squat":   {"unit": "reps",    "count": (8, 20),  "sets": (2, 4), "rest": (20, 90),  "weight_capable": True,  "weight_range": (0, 15)},
    "lunge":   {"unit": "reps",    "count": (8, 20),  "sets": (2, 3), "rest": (30, 120), "weight_capable": True,  "weight_range": (0, 10)},
    "pushup":  {"unit": "reps",    "count": (8, 15),  "sets": (2, 4), "rest": (20, 90),  "weight_capable": False, "weight_range": (0, 0)},
}


def find_abnormal(metrics: dict) -> list:
    """정상 범위 벗어난 항목을 편차순 정렬해 반환. (기존과 동일)"""
    abnormal = []
    for key, (label, lo, hi, direction) in REF_RANGES.items():
        if key not in metrics:
            continue
        v = metrics[key]
        if lo <= v <= hi:
            continue
        dev = lo - v if v < lo else v - hi
        abnormal.append({
            "label": label, "value": v,
            "range": f"{lo}~{hi}", "deviation": round(dev, 1),
        })
    abnormal.sort(key=lambda x: x["deviation"], reverse=True)
    return abnormal


# ================================================
# 호출 1: 스트레칭 추천 (자유 형식, 표시 전용)
# ================================================
def build_stretch_prompt(abnormal: list) -> str:
    lines = [
        f"- {a['label']}: measured {a['value']} "
        f"(normal {a['range']}, deviation {a['deviation']})"
        for a in abnormal
    ]
    body = "\n".join(lines)

    return f"""You are a posture-correction coach.
Below are posture issues judged abnormal, largest deviation first.

{body}

For each issue, recommend a suitable stretch (you may choose any stretch freely).
Write in KOREAN, plain and short. Do NOT give medical diagnoses.

Respond ONLY in this JSON format:
{{
  "stretches": [
    {{"name": "스트레칭 이름", "reason": "짧은 한국어 설명"}}
  ]
}}"""


def recommend_stretches(abnormal: list) -> list:
    """스트레칭 추천 리스트 반환. 표시 전용. 실패 시 빈 리스트."""
    if not abnormal:
        return []
    prompt = build_stretch_prompt(abnormal)
    raw = ask_ollama(prompt)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("[stretch] JSON 파싱 실패:", raw)
        return []
    return data.get("stretches", [])


# ================================================
# 호출 2: 운동 처방 (3종 내, workout_sel 연동)
# ================================================
GOAL_DESC = {
    "posture":   "posture correction (low intensity, focus on control)",
    "strength":  "strength building (higher load, fewer reps)",
    "endurance": "endurance (more reps/longer duration)",
}


def _age_factor(age: int = None) -> float:
    """
    나이에 따른 강도 감쇠 계수 (1.0 = 감쇠 없음, 0.4 = 하한).

    - 18~54세: 표준 성인 구간, 감쇠 없음 (factor = 1.0)
    - 18세 미만: 나이가 어릴수록 완만하게 감소 (성장기 안전 고려, 조금 더 가파른 기울기)
    - 55세 이상: 나이가 많을수록 완만하게 감소 (18세 미만보다 더 완만한 기울기)

    기존에는 나이가 범위를 벗어나면 즉시 pos를 0.3으로 캡(cap)하는 방식이라
    54세와 55세 사이에서 강도가 급격히 끊기는 문제가 있었음.
    거리(distance)에 비례해 계수를 서서히 낮춰서 경계에서도 자연스럽게 이어지도록 함.
    """
    if age is None:
        return 1.0
    if 18 <= age < 55:
        return 1.0
    if age < 18:
        distance = 18 - age
        return max(0.4, 1.0 - distance * 0.06)
    distance = age - 54
    return max(0.4, 1.0 - distance * 0.02)


def _target_position(level: str, goal: str, age: int = None) -> float:
    """레벨/목표/나이를 반영한 0~1 위치. 높을수록 안전 범위의 상단(더 높은 강도) 쪽.

    레벨(기본 위치) -> 목표(가산 보정) -> 나이(비례 감쇠) 순서로 계산해서,
    세 요인이 순차적으로 자연스럽게 합성되도록 함.
    """
    pos = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8}.get(level, 0.5)

    if goal == "endurance":
        pos += 0.15  # 지구력 목표면 reps를 조금 더

    pos *= _age_factor(age)  # 나이 요인은 비례 감쇠로 반영 (급격한 캡 대신)

    return min(max(pos, 0.0), 1.0)


def _rest_position(goal: str) -> float:
    """휴식 시간은 reps/sets와 반대 방향(지구력=짧게, 근력=길게)이라 별도 위치로 계산."""
    if goal == "endurance":
        return 0.2
    if goal == "strength":
        return 0.8
    return 0.5


def _target_value(lo, hi, pos: float) -> int:
    return round(lo + (hi - lo) * pos)


def _suggest_weight(name: str, level: str, goal: str, age: int = None) -> str:
    """권장 중량을 코드에서 결정론적으로 계산 (LLM 판단에 맡기지 않음).
    count/sets/rest와 동일하게 weight_range + pos 보간 공식을 그대로 재사용해서,
    별도의 임의 구간(예: 0.3/0.5/0.7 등급 나누기) 없이 일관되게 산출한다."""
    r = EXERCISE_POOL[name]
    if goal != "strength" or not r["weight_capable"]:
        return ""

    pos = _target_position(level, goal, age)
    w = _target_value(*r["weight_range"], pos)

    if w <= 1:
        return "체중 유지, 무게 추가 없음"
    return f"덤벨 {w}kg"


def build_workout_prompt(goal: str, level: str, age: int = None) -> str:
    goal_text = GOAL_DESC.get(goal, "general fitness")

    # 레벨/나이/목표를 코드에서 미리 반영해 이미 좁혀진 목표 수치를 계산.
    # LLM에게 범위를 텍스트로 설명하고 계산을 맡기지 않아 프롬프트가 짧고 결과가 안정적이다.
    pos = _target_position(level, goal, age)
    rest_pos = _rest_position(goal)

    # weight도 count/sets/rest와 마찬가지로 코드에서 미리 계산 (LLM이 판단하지 않음)
    exercises_text = "\n".join(
        f"- {name}: target {_target_value(*r['count'], pos)} {r['unit']}, "
        f"{_target_value(*r['sets'], pos)} sets, "
        f"rest {_target_value(*r['rest'], rest_pos)}s between sets"
        + (f", weight \"{_suggest_weight(name, level, goal, age)}\"" if goal == "strength" else "")
        for name, r in EXERCISE_POOL.items()
    )

    if goal == "strength":
        goal_instruction = (
            '- Use the "weight" value shown above for each exercise exactly as given — '
            'do not change or recalculate it.\n'
        )
    else:
        goal_instruction = '- Leave the "weight" field as an empty string "" (no added weight for this goal).\n'

    return f"""You are an exercise coach.
User goal: {goal_text}

Exercises with pre-computed targets (already tailored to the user's level/age/goal):
{exercises_text}

Instructions:
- You MUST recommend EVERY SINGLE exercise listed above — squat, lunge, AND pushup — with no exceptions.
  Never omit one, never return fewer than 3 items, and NEVER return an empty "recommendations" list.
- Use the target count/sets/rest_seconds shown above for each exercise as-is — do not recalculate them.
- Use the unit shown above for each exercise.
{goal_instruction}- Set "priority" as the order to perform (1 = first), ordering them to fit the goal.
- Give a one-sentence reason IN KOREAN (short).
- Do NOT give medical diagnoses.

Respond ONLY in this JSON format. "recommendations" MUST contain exactly 3 items — one each for
squat, lunge, and pushup. A partial or empty list is INVALID:
{{
  "recommendations": [
    {{"exercise": "squat", "priority": ..., "count": ..., "unit": "reps", "sets": ..., "rest_seconds": ..., "weight": "...", "reason": "..."}},
    {{"exercise": "lunge", "priority": ..., "count": ..., "unit": "reps", "sets": ..., "rest_seconds": ..., "weight": "...", "reason": "..."}},
    {{"exercise": "pushup", "priority": ..., "count": ..., "unit": "reps", "sets": ..., "rest_seconds": ..., "weight": "", "reason": "..."}}
  ]
}}"""


def _clamp(v, lo, hi):
    return max(lo, min(int(v), hi))


def prescribe_workouts(goal: str, level: str, age: int = None) -> dict:
    """
    운동 처방을 workout_sel이 먹을 형식으로 반환.
    반환: {"recommendations": [...]}  (횟수는 안전 범위로 clamp)
    """
    prompt = build_workout_prompt(goal, level, age)
    raw = ask_ollama(prompt)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("[workout] JSON 파싱 실패:", raw)
        return {"recommendations": []}

    cleaned = []
    for rec in data.get("recommendations", []):
        ex = rec.get("exercise")
        if ex not in EXERCISE_POOL:      # 목록 밖 운동은 버림
            continue
        r = EXERCISE_POOL[ex]
        rec["count"] = _clamp(rec.get("count", r["count"][0]), *r["count"])
        rec["sets"]  = _clamp(rec.get("sets",  r["sets"][0]),  *r["sets"])
        rec["rest_seconds"] = _clamp(rec.get("rest_seconds", r["rest"][0]), *r["rest"])  # 추가
        rec["unit"]  = r["unit"]
        # weight는 LLM 응답을 신뢰하지 않고, count/sets/rest와 동일하게 코드 계산값으로 강제 덮어씀.
        # (LLM이 프롬프트를 무시하고 다른 값을 내더라도 최종 결과는 항상 일관되게 유지)
        rec["weight"] = _suggest_weight(ex, level, goal, age)
        cleaned.append(rec)

    # priority 순 정렬해서 반환
    cleaned.sort(key=lambda x: x.get("priority", 99))
    return {"recommendations": cleaned}


def ask_ollama(prompt: str) -> str:
    print("Asking to ollama...")
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        keep_alive="0",
        options={"temperature": 0, "num_ctx": 2048},
    )
    return response["message"]["content"]
