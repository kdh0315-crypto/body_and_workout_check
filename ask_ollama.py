import json
import ollama

# 각도 키 -> (라벨, 정상 최소, 정상 최대, 방향)
#   direction:
#     "abs"  = 0에서 멀수록 심함 (좌우 기울기 등)
#     "high" = 클수록 심함 (범위 위로 벗어남이 문제)
#     "low"  = 작을수록 심함 (범위 아래로 벗어남이 문제)
REF_RANGES = {
    "shoulder_tilt":     ("어깨 좌우 기울기",           0,  2, "abs"),
    "knee_valgus":       ("무릎 정렬 편차(좌/우 평균)",  0,  5, "high"),
    "forward_head":      ("거북목 각도(귀-어깨-엉덩이)", 55, 65, "low"),
    "round_shoulder":    ("라운드숄더 각도",            0, 10, "high"),
    "thoracic_kyphosis": ("흉추 후만 각도",            30, 40, "high"),
    "pelvic_tilt_ant":   ("골반 전방경사 각도",          5, 10, "high"),
}

AVAILABLE_EXERCISES = "스쿼트, 플랭크, W/Y레이즈, 승모 스트레칭"


def find_abnormal(metrics: dict) -> list:
    """정상 범위를 벗어난 항목만 골라 '벗어난 정도'와 함께 반환 (심한 순 정렬)."""
    abnormal = []
    for key, (label, lo, hi, direction) in REF_RANGES.items():
        if key not in metrics:
            continue
        v = metrics[key]

        # 정상 범위 안이면 건너뜀
        if lo <= v <= hi:
            continue

        # 벗어난 정도(deviation) 계산
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

    # 많이 벗어난 순으로 정렬
    abnormal.sort(key=lambda x: x["deviation"], reverse=True)
    return abnormal


def build_prompt(abnormal: list) -> str:
    """이미 이상으로 걸러진 항목만 LLM에 넘긴다. 판정은 시키지 않는다."""
    lines = [
        f"- {a['label']}: 측정 {a['value']} "
        f"(정상 {a['range']}, 벗어난 정도 {a['deviation']})"
        for a in abnormal
    ]
    body = "\n".join(lines)

    return f"""당신은 자세 교정 운동 코치다.
아래는 이미 '정상 범위를 벗어났다고 판정된' 항목들이다 (벗어난 정도가 큰 순).

{body}

추천 가능한 운동: {AVAILABLE_EXERCISES}

지침:
- 위 각 이상 항목에 가장 알맞은 운동을 '추천 가능한 운동' 중에서만 골라라.
- 목록에 없는(=정상인) 문제는 언급하지 마라.
- 벗어난 정도가 큰 항목일수록 우선순위(priority)를 높게(숫자를 작게) 준다.
- 우선순위가 가장 높은 2개만 추천하라.
- 같은 운동이 여러 문제에 해당하면 한 번만 추천한다.
- 각 운동을 왜 골랐는지 해당 측정값을 근거로 한 문장씩 설명한다.
- 근거는 20자 이내로 짧게 설명하라.
- 의료 진단은 하지 말 것.

아래 JSON 형식으로만 답하라. 다른 말은 붙이지 마라:
{{
  "recommendations": [
    {{"exercise": "운동명", "priority": 1, "reason": "측정값 근거 한 문장"}}
  ]
}}"""


def ask_ollama(prompt: str) -> str:
    print("Asking to ollama...")
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
        format="json",       # JSON 형식 강제
        keep_alive="5m",     # 모델 메모리 유지 (재로딩 지연 방지)
        options={
            "temperature": 0.3,   # 재현성 위해 낮게
            "num_ctx": 2048,      # KV 캐시 절약 (Orin 메모리 고려)
        },
    )
    return response["message"]["content"]


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