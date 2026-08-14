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
EXERCISE_POOL = {
    "squat":   {"unit": "reps",    "count": (8, 20),  "sets": (2, 4), "rest": (20, 90)},
    "lunge":   {"unit": "reps",    "count": (8, 20),  "sets": (2, 3), "rest": (30, 120)},
    "pushup":  {"unit": "reps",    "count": (8, 15),  "sets": (2, 4), "rest": (20, 90)},
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


LEVEL_RANGE_POSITION = {
    "beginner":     "the LOWER third",
    "intermediate": "the MIDDLE third",
    "advanced":     "the UPPER third",
}


def _age_instruction(age) -> str:
    if age is None:
        return ""
    if age < 18 or age >= 55:
        return (
            f"- User age is {age}, outside the standard adult range: reduce intensity further — "
            "shift count/sets toward the LOWER end of the safe range regardless of level.\n"
        )
    return f"- User age is {age}: no extra age-based reduction beyond the level guidance below.\n"


def build_workout_prompt(goal: str, level: str, age: int = None) -> str:
    goal_text = GOAL_DESC.get(goal, "general fitness")
    # 범위 안내에 rest 추가
    ranges_text = "\n".join(
        f"- {name}: {r['count'][0]}-{r['count'][1]} {r['unit']}, "
        f"{r['sets'][0]}-{r['sets'][1]} sets, "
        f"rest {r['rest'][0]}-{r['rest'][1]} seconds between sets"
        for name, r in EXERCISE_POOL.items()
    )

    level_position = LEVEL_RANGE_POSITION.get(level, "the MIDDLE third")
    age_instruction = _age_instruction(age)

    if goal == "strength":
        goal_instruction = (
            '- Since goal is strength building: for EACH exercise, also suggest an appropriate added '
            'weight (dumbbell/vest, in kg) in the "weight" field, based on the user\'s age and level '
            '(e.g. "덤벨 5kg", or "체중 유지, 무게 추가 없음" for beginners/reduced-intensity users). '
            "Keep it conservative and safe.\n"
        )
    elif goal == "endurance":
        goal_instruction = (
            '- Since goal is endurance: push "count" (reps) toward the UPPER end of each exercise\'s '
            'safe range, keep sets moderate, and leave "weight" as an empty string "".\n'
        )
    else:
        goal_instruction = '- Leave the "weight" field as an empty string "" (no added weight for this goal).\n'

    return f"""You are an exercise coach.
User goal: {goal_text}
User level: {level}
User age: {age if age is not None else "unknown"}

Available exercises and SAFE ranges (do not exceed):
{ranges_text}

Instructions:
- You MUST recommend EVERY SINGLE exercise listed above — squat, lunge, AND pushup — with no exceptions.
  Never omit one, never return fewer than 3 items, and NEVER return an empty "recommendations" list.
- For each exercise, assign count and sets WITHIN its safe range, matched to the user's goal.
- Base count/sets primarily on level: use {level_position} of each safe range as a starting point.
{age_instruction}{goal_instruction}- Set "priority" as the order to perform (1 = first), ordering them to fit the goal.
- Use the unit shown above for each exercise.
- Assign rest_seconds between sets WITHIN the safe range (shorter for endurance, longer for strength).
- Give a one-sentence reason IN KOREAN (short).
- Do NOT give medical diagnoses.

Respond ONLY in this JSON format. "recommendations" MUST contain exactly 3 items — one each for
squat, lunge, and pushup. A partial or empty list is INVALID:
{{
  "recommendations": [
    {{"exercise": "squat", "priority": 1, "count": 12, "unit": "reps", "sets": 3, "rest_seconds": 45, "weight": "", "reason": "..."}},
    {{"exercise": "lunge", "priority": 2, "count": 12, "unit": "reps", "sets": 3, "rest_seconds": 30, "weight": "", "reason": "..."}},
    {{"exercise": "pushup", "priority": 3, "count": 12, "unit": "reps", "sets": 3, "rest_seconds": 60, "weight": "", "reason": "..."}}
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
        # 근력 강화 목표일 때만 중량 추천을 노출 (다른 목표는 LLM이 뭐라 적었든 무시)
        rec["weight"] = rec.get("weight", "") if goal == "strength" else ""
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