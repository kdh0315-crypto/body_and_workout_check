import json
import ollama

# angle key -> (label, normal_min, normal_max, direction)
#   direction:
#     "abs"  = worse the farther from 0 (e.g., left-right tilt)
#     "high" = worse when higher (deviation above range is the problem)
#     "low"  = worse when lower (deviation below range is the problem)
REF_RANGES = {
    "shoulder_tilt":     ("Shoulder lateral tilt",              0,  2, "abs"),
    "knee_valgus":       ("Knee alignment deviation (L/R avg)", 0,  5, "high"),
    "forward_head":      ("Forward head angle (ear-shoulder-hip)", 55, 65, "low"),
    "round_shoulder":    ("Round shoulder angle",               0, 10, "high"),
    "thoracic_kyphosis": ("Thoracic kyphosis angle",           30, 40, "high"),
    "pelvic_tilt_ant":   ("Anterior pelvic tilt angle",         5, 10, "high"),
}

AVAILABLE_EXERCISES = "squat, plank, biceps curl"


def find_abnormal(metrics: dict) -> list:
    """
    find abnormal data from normal range
    sorting according to deviation
    """
    abnormal = []
    for key, (label, lo, hi, direction) in REF_RANGES.items():
        if key not in metrics:
            continue
        v = metrics[key]

        # pass if in normal range
        if lo <= v <= hi:
            continue

        # calculate deviation
        if v < lo:
            dev = lo - v
        else:  # v > hi
            dev = v - hi

        abnormal.append({
            "label": label,
            "value": v,
            "range": f"{lo}~{hi}",
            "deviation": round(dev, 1),
        })

    # sort according to deviation value
    abnormal.sort(key=lambda x: x["deviation"], reverse=True)
    return abnormal


def build_prompt(abnormal: list) -> str:
    """make prompt to ollama can recommend workout"""
    lines = [
        f"- {a['label']}: measured {a['value']} "
        f"(normal {a['range']}, deviation {a['deviation']})"
        for a in abnormal
    ]
    body = "\n".join(lines)

    return f"""You are a posture-correction exercise coach.
Below are items ALREADY judged to be outside the normal range,
sorted by how far they deviate (largest first).

{body}

Available exercises: {AVAILABLE_EXERCISES}

Instructions:
- For each abnormal item above, pick the most suitable exercise ONLY from "Available exercises".
- Do NOT mention any problem that is not in the list (i.e., normal items).
- The larger the deviation, the higher the priority (smaller number = higher priority).
- Recommend ONLY the 2 highest-priority items.
- If one exercise addresses multiple problems, recommend it only once.
- Give a one-sentence reason for each choice, grounded in its measured value.
- Keep each reason within 20 characters.
- Do NOT give medical diagnoses.

Respond ONLY in the following JSON format. Do not add any other text:
{{
  "recommendations": [
    {{"exercise": "exercise name", "priority": 1, "reason": "short reason"}}
  ]
}}"""


# ================================================
# Send prompt to ollama
# ================================================
def ask_ollama(prompt: str) -> str:
    print("Asking to ollama...")
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        format="json",          # force JSON file platform
        keep_alive="0",         # do not maintain memory allocate
        options={
            "temperature": 0,   # make response stick
            "num_ctx": 2048,    # set KV cache token
        }
    )
    return response["message"]["content"]


# ================================================
# To print data at GUI platform
# ================================================
def recommend_exercise(metrics: dict):
    # 1) 코드가 먼저 정상/이상을 판정
    abnormal = find_abnormal(metrics)

    if not abnormal:
        print("=== 모든 측정값이 정상 범위입니다. 추천 없음. ===")
        return

    print(f"[이상 항목 {len(abnormal)}개 감지] "
          + ", ".join(a["label"] for a in abnormal))

    # 2) 이상 항목만 LLM에 넘겨 매칭/설명/우선순위 생성
    prompt = build_prompt(abnormal)
    raw = ask_ollama(prompt)

    # 3) 결과 파싱 및 출력
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("[경고] JSON 파싱 실패, 원본 출력:")
        print(raw)
        return

    recs = sorted(data.get("recommendations", []),
                  key=lambda r: r.get("priority", 99))
    print("=== 추천 운동 ===")
    for r in recs:
        print(f"[{r.get('priority')}] {r.get('exercise')}"
              f"  -  {r.get('reason')}")