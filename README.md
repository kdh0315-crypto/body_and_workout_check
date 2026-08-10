# 자세 분석 및 운동 추천 시스템

MediaPipe Pose 기반으로 정면/측면 촬영 이미지에서 자세를 분석하고, 자세 편차를 검출한 뒤 교정 운동을 추천하는 시스템. NVIDIA Jetson Orin Nano 배포를 목표로 개발

## 개요

- **입력**: 웹캠으로 정면 / 측면 자세 2장 순차 촬영
- **분석**: MediaPipe Pose 랜드마크 추출 → 자세 지표(각도) 계산 → 임계값 비교로 편차 검출
- **추천**: 사전 필터링된 이상 항목만 로컬 LLM(Ollama, gemma3:4b)에 전달 → 운동 매칭·우선순위·설명 생성
- **운동 진행**: FSM 기반 `workout_sel`이 우선순위 순으로 운동을 순차 배출

### 하이브리드 아키텍처

객관적 각도 임계값 비교·편차 검출은 **Python 코드**(`find_abnormal`)가 담당하고, 운동 매칭·우선순위·설명 생성은 **로컬 Ollama**가 담당한다. LLM에 원시(raw) 각도 값을 직접 해석시키면 성능이 나빠져, 역할을 명확히 분리했다.

```
사용자 정보 입력 (UserInfoForm)
        │
        ▼
카메라 촬영 (CameraView + CaptureForm) ── 정면 → 측면 순차 캡처
        │
        ▼
MediaPipe 랜드마크 추출 (save_with_pose / extract_landmarks)
        │
        ▼
관절 좌표 추출 (get_pose_point_norm)
        │
        ▼
각도 지표 계산 (upper_body / lower_body)
   FHA · FSA · shoulder tilt · thoracic kyphosis · pelvic tilt · knee valgus
        │
        ▼
find_abnormal() — REF_RANGES 대비 편차 계산·심각도 정렬
        │
        ▼ (이상 항목만 전달)
Ollama (gemma3:4b) — 운동 매칭·우선순위·설명 (JSON)
        │
        ▼
workout_sel — 우선순위 순 운동 진행 (idle → work → idle)
```

## 파일 구조

```
project/
├── main.py                     # 진입점: QApplication 실행, UserInfoForm 표시
├── README.md
├── .gitignore
│
├── gui/
│   ├── gui.py                  # UserInfoForm(정보 입력) + CameraView(촬영·분석)
│   └── gui_style.py            # STYLE: 다크 테마 QSS 문자열
│
├── module/
│   ├── basic_fn.py             # 공용 계산 유틸 (3점 각도, 픽셀 변환, 중간점)
│   ├── mediapipe_op.py         # CaptureForm + MediaPipe Tasks 랜드마커·스켈레톤·좌표 추출
│   ├── upper_body.py           # 상체 각도 추출: FHA, FSA, shoulder tilt, thoracic kyphosis
│   ├── lower_body.py           # 하체 각도 추출: anterior pelvic tilt, knee valgus
│   ├── ollama_op.py            # REF_RANGES, find_abnormal, 프롬프트, Ollama 호출
│   ├── workout_sel.py          # workout_sel FSM - 추천된 운동 순차적으로 실행
│   ├── test_fn.py              # Test & Debug용 함수 모음
│   └── models/
│       └── pose_landmarker_full.task   # MediaPipe Pose Landmarker 모델
│
└── docs/
    └── Effect_of_Rounded_and_Hunched_Shoulder_Postures...pdf  # RSA/HSA 참고 논문
```

> 참고: `main.py`는 `from module.* import *` 와 `from gui.gui import *` 형태로 import하므로,
> `gui/`·`module/`은 패키지로 인식되도록 실행 위치(루트)에서 실행해야 한다.
> MediaPipe 모델 경로는 `mediapipe_op.py`에 `module/models/pose_landmarker_full.task`로 하드코딩되어 있다.

## 자세 지표

| 지표 | 함수 | 정상 판정 | 파일 |
|------|------|-----------|------|
| FHA (forward head) | `calculate_fha` / `classify_fha` | 50~60° normal, 이하 mild/moderate/severe | upper_body.py |
| FSA (forward shoulder) | `calculate_fsa` / `classify_fsa` | < 52° normal (Thigpen 기준) | upper_body.py |
| Shoulder tilt | `calculate_shoulder_tilt` / `classify_shoulder_tilt` | \|각도\| ≤ 2.5° normal | upper_body.py |
| Thoracic kyphosis | `calculate_thoracic_kyphosis` / `classify_thoracic_kyphosis` | 20~40° normal | upper_body.py |
| Anterior pelvic tilt | `calculate_pelvic_tilt_ant` / `classify_pelvic_tilt_ant` | 8~18° normal | lower_body.py |
| Knee valgus (FPPA) | `calculate_knee_valgus_angle` | (분류기 미정, 임계값 캘리브레이션 필요) | lower_body.py |

- 각도 계산은 정규화 좌표를 픽셀로 변환한 뒤 2D `(x, y)` 평면에서 수행 (z축 미사용)
- `lower_body.py`의 벡터 각도(`calculate_vector_angle`)는 `np.clip(..., -1.0, 1.0)`으로 `arccos` 도메인 오류 방지

## 주요 컴포넌트

### GUI (`gui/gui.py`)
- **UserInfoForm**: 나이 / 체력 수준(초급·중급·고급) / 목표(자세교정·근력·지구력) 입력 → `{age, level, goal}` dict 생성 후 CameraView로 전환
- **CameraView**: `QTimer`(30ms ≈ 33fps) 카메라 루프, OpenCV(BGR) → `QImage`/`QPixmap`(RGB) 변환. '촬영' 버튼(`on_capture`)으로 CaptureForm에 프레임 전달
  - 촬영 완료 시 `save_with_pose`로 스켈레톤 저장 + `get_pose_point_norm`으로 정면/측면 관절 좌표 추출 (현재는 콘솔 출력까지)
  - 감지 실패 시 `_restart()`로 재촬영
- **다크 테마**: `gui_style.py`의 `STYLE` 문자열을 `main.py`에서 `app.setStyleSheet(STYLE)`로 전역 적용

### CaptureForm (`module/mediapipe_op.py`)
- 정면 → 측면 순서로 순차 캡처
- `key == ord('c')` + `key_released` 플래그로 정면/측면 동시 촬영 방지(디바운싱)
- GUI 버튼 방식에서는 `key_released = True`를 세팅한 뒤 `capture_form(ord('c'), frame)` 호출
- 측면까지 완료되면 `(front_img, side_img)` 반환하고 `done = True`

### MediaPipe (`module/mediapipe_op.py`)
- **Tasks API** 사용 (`vision.PoseLandmarker`, `pose_landmarker_full.task`), IMAGE / VIDEO 러닝 모드 둘 다 초기화
- 스켈레톤 드로잉은 legacy `mp.solutions.drawing_utils` + `POSE_CONNECTIONS` 사용 (Tasks 결과를 `NormalizedLandmarkList` proto로 변환)
- `KEY_LANDMARKS`로 코·귀·어깨·엉덩이·무릎·발목만 이름 기반 추출
- 좌표 추출: `get_landmark_pixels`(픽셀), `get_pose_point`(이름별 픽셀), `get_pose_point_norm`(이름별 정규화, `with_vis` 옵션)

### 이상 검출 & 추천 (`module/ollama_op.py`)
- **REF_RANGES**: 지표별 `(label, normal_min, normal_max, direction)`. direction은 `abs`/`high`/`low`
- **find_abnormal**: 정상 범위 밖 항목만 추려 편차 크기 내림차순 정렬
- **build_prompt**: 영어 지시 프롬프트. 이상 항목만 나열, 상위 2개만 추천, 각 이유는 짧게, JSON만 출력 요구
- **ask_ollama**: `model="gemma3:4b"`, `format="json"`, `temperature=0`, `num_ctx=2048`, `keep_alive="0"`
- **recommend_exercise**: 판정 → 프롬프트 → 호출 → JSON 파싱 → 우선순위 정렬 출력
- 현재 `AVAILABLE_EXERCISES = "squat, plank, biceps curl"`

### workout_sel (`module/workout_sel.py`)
- 상태: `idle` / `work`
- `load_workout(ollama_response)`: `recommendations`를 priority 순 정렬, 있으면 `work`로 전이
- `current_workout()`: 현재 운동 반환(없으면 None)
- `next_workout(work_done)`: `work_done=True`일 때만 `work_cnt` 증가, 모두 끝나면 `idle` 복귀
- 테스트: `module/test_fn.py`의 `test_workout_sel`, `test_edge_cases`

## 개발 환경

- **하드웨어**: NVIDIA Jetson Orin Nano (aarch64) 배포 / x86_64 PC 개발
- **핵심 라이브러리**: MediaPipe, OpenCV, NumPy, PySide6, ollama (공식 파이썬 라이브러리)
- **LLM**: 로컬 Ollama + gemma3:4b

## 실행

```bash
# 프로젝트 루트에서
python main.py
```

1. 정보 입력 폼에서 나이·체력 수준·목표 선택 후 제출
2. 카메라 화면에서 정면 자세 → '촬영', 측면 자세 → '촬영'
3. 스켈레톤 저장 및 관절 좌표 추출 (콘솔 확인)

## 진행 상황

**완료**
- 자세 지표 계산 함수 (FHA/FSA/shoulder tilt/kyphosis/pelvic tilt/knee valgus)
- `REF_RANGES` / `find_abnormal` / Ollama 프롬프트·호출
- `workout_sel` FSM (+ 단위 테스트)
- CaptureForm (디바운싱) + MediaPipe Tasks 파이프라인
- PySide6 다크 테마 GUI (정보 입력 → 촬영 → 좌표 추출)

**진행 예정**
- 추출된 관절 좌표를 각도 계산 함수에 연결 → `metrics` dict 구성 → `recommend_exercise` 호출까지 파이프라인 결선 (현재 좌표 추출 후 콘솔 출력에서 끊김)
- REF_RANGES 지표 키와 실제 계산 함수 출력 이름 정합 (예: `forward_head`↔FHA, `round_shoulder`↔FSA 매핑)
- `AVAILABLE_EXERCISES`를 실제 목표 운동(W/Y raise, 승모근 스트레칭 등)으로 확장
- Knee valgus 임계값 캘리브레이션 (MediaPipe 무릎 각도 오차, 정상군 데이터 기반 보정 필요)
- XGBoost 분류기를 Layer 2 오류 분류 모델로 도입 검토

## 설계 원칙 / 학습

- **z축 미사용**: MediaPipe z 추정 정확도가 낮아 카메라 평면별 2D `(x, y)`만 사용
- **`np.clip` 필수**: dot product 각도 계산 시 `arccos` 부동소수점 도메인 오류 방지
- **LLM 역할 제한**: 잘못된 추천을 막기 위해 사전 필터링된 이상 항목만 전달, 정상 항목은 언급 금지
- **Shoulder tilt / pelvic tilt**: 좌우 자연 비대칭·랜드마크 추정 한계로 임계값이 민감함 (참고용 성격이 강함)
- **참고 논문**: RSA/HSA 정상군 값을 지표 정상 범위 산정에 활용