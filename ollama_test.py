import json
import ollama
from ask_ollama import *


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

    print(">>> 감지된 이상 항목\n")
    for a in find_abnormal(sample_metrics):
        print(f"  {a['label']}: {a['value']} "
              f"(정상 {a['range']}, 벗어남 {a['deviation']})")

    print("\n>>> Ollama 호출 결과\n")
    recommend_exercise(sample_metrics)