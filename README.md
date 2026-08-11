# Body and Workout Check — Integration Status

> 2026-08-11 기준 통합본  
> Branch: `integration_input_mediapipe`

## 1. 현재 목표

정면/측면 이미지를 입력받아 MediaPipe 기반 Body Check를 수행하고,
FHA는 Rule + TensorRT AI + Fusion으로 최종 판정한 뒤,
전체 자세 결과를 `find_abnormal()` → Ollama로 전달하여
스트레칭/운동 추천을 생성한다.

```text
Front / Side Image
        ↓
MediaPipe Pose
        ↓
calculate_all_features()
        ↓
Body Check Metrics
        │
        ├─ Shoulder Tilt
        ├─ Knee Valgus
        ├─ FHA (Rule + TensorRT AI + Fusion)
        ├─ FSA
        ├─ Thoracic Kyphosis
        └─ Pelvic Tilt
        ↓
features_to_metrics()
        ↓
find_abnormal()
        ↓
LLMWorker
        ↓
Ollama
        ↓
Stretch / Workout Recommendation
```

## 2. 현재 주요 파일

```text
body_and_workout_check/
├── gui/
│   └── gui.py
├── module/
│   ├── cal_angle.py
│   ├── upper_body.py
│   ├── lower_body.py
│   ├── fha_integration.py
│   └── ollama_op.py
└── body_static_pose/
    ├── fha_ai.py
    └── runtime_models/
        └── fha_mobilenet.engine
```

## 3. FHA 통합 구조

현재 FHA는 오른쪽 측면 기준으로 고정한다.

```text
RIGHT_EAR + RIGHT_SHOULDER
```

Rule 기준:

```text
< 32°          → RULE_NORMAL
32° ~ < 36°    → RULE_BORDERLINE
>= 36°         → RULE_ABNORMAL
```

AI 기준:

```text
score < 0.55            → AI_NORMAL
0.55 <= score < 0.75    → AI_BORDERLINE
score >= 0.75           → AI_ABNORMAL
```

Fusion 기준:

```text
RULE_ABNORMAL → FUSION_ABNORMAL
AI_ABNORMAL   → FUSION_ABNORMAL
RULE_NORMAL + AI_NORMAL → FUSION_NORMAL
그 외 → FUSION_BORDERLINE
```

## 4. FSA

FHA와 측면 기준을 맞추기 위해 오른쪽 어깨를 사용한다.

```text
NECK_CENTER + RIGHT_SHOULDER
```

FSA 기준은 52°로 통일했다.

```text
FSA < 52°  → normal
FSA >= 52° → abnormal
```

`module/upper_body.py`와 `module/ollama_op.py`의 기준을 일치시켰다.

## 5. Pelvic Tilt

`calculate_all_features()`의 계산 key:

```text
pelvic_tilt_ant_deg
```

최종 metrics key:

```text
pelvic_tilt_ant
```

`features_to_metrics()`에 아래 mapping을 추가했다.

```python
"pelvic_tilt_ant": side_features.get("pelvic_tilt_ant_deg"),
```

실제 테스트에서 `side_features`의 Pelvic 값이 `metrics`까지 전달되는 것을 확인했다.

## 6. 최종 Metrics 구조

```python
metrics = {
    "shoulder_tilt": ...,
    "knee_valgus": ...,
    "forward_head": ...,
    "round_shoulder": ...,
    "thoracic_kyphosis": ...,
    "pelvic_tilt_ant": ...,
    "forward_head_fusion": ...,
}
```

측정 실패로 `None`인 값은 최종 metrics에서 제외한다.

## 7. 실제 검증 완료

- [x] 사용자 정보 입력
- [x] Front Image Capture
- [x] Side Image Capture
- [x] MediaPipe landmark 생성
- [x] `front_features` 생성
- [x] `side_features` 생성
- [x] FHA angle 계산
- [x] FHA Rule 판정
- [x] TensorRT FHA AI 추론
- [x] FHA AI score/result
- [x] FHA Fusion
- [x] FHA Fusion → metrics
- [x] Shoulder Tilt 계산 및 metrics 전달
- [x] Left Knee Alignment 계산
- [x] Right Knee Alignment 계산
- [x] Left Knee Valgus 계산
- [x] Right Knee Valgus 계산
- [x] Knee Valgus 평균 → metrics
- [x] Thoracic Kyphosis 계산 및 metrics 전달
- [x] FSA 실제 숫자 출력
- [x] FSA → `round_shoulder` 전달
- [x] Pelvic Tilt → `pelvic_tilt_ant` 연결
- [x] Pelvic Tilt 실제 숫자 → metrics 전달 확인
- [x] `find_abnormal()` 실행
- [x] Ollama 호출
- [x] Stretch 추천 반환
- [x] Workout 추천 반환

## 8. 실제 검증 예

```text
FHA Rule: 15.8269 / RULE_NORMAL
FHA AI: 0.3945 / AI_NORMAL
FHA Fusion: FUSION_NORMAL
```

FSA:

```text
side_features:
'fsa_deg': 80.98559415013713
```

Pelvic Tilt:

```text
side_features:
'pelvic_tilt_ant_deg': 5.970216756103923
```

```text
metrics:
'pelvic_tilt_ant': 5.970216756103923
```

## 9. 현재 남은 작업

- [ ] FSA와 Pelvic Tilt가 동시에 숫자로 나오는 측면 이미지로 최종 1회 재확인
- [ ] 전체 수정 파일 `py_compile`
- [ ] `git diff` 최종 확인
- [ ] 테스트용 `front_features`, `side_features` print 유지/제거 결정
- [ ] 팀 공유용 commit
- [ ] merge 전 충돌 확인

## 10. 추후 개선사항

현재 통합 범위에서는 제외한다.

- [ ] 측면 LEFT / RIGHT 자동 판별
- [ ] landmark visibility 비교
- [ ] 더 신뢰도 높은 EAR / SHOULDER 자동 선택
- [ ] FHA / FSA 측면 방향 자동 통일
- [ ] 반대 방향 촬영 fallback
- [ ] landmark visibility 부족 시 재촬영 안내
- [ ] TensorRT engine device compatibility 관리

## 11. Git 공유 시 주의

작업 디렉터리에는 모델, ZIP, 테스트 이미지, dataset 등이 있으므로
`git add .` 사용을 피하고 필요한 파일만 명시적으로 stage한다.

```bash
git add \
gui/gui.py \
module/fha_integration.py \
module/cal_angle.py \
module/upper_body.py \
module/ollama_op.py \
README.md
```

`main.py`는 기존 474줄 구현이 GUI entry point로 교체된 상태이므로,
팀 공유 전에 별도 검토 후 포함 여부를 결정한다.
