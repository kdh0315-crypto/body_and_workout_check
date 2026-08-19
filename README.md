# Body & Workout Check

MediaPipe 포즈 추정과 LLM(Ollama)을 결합한 **자세 분석 + 운동 처방 + 실시간 운동 자세 판별** 통합 데스크톱 애플리케이션입니다. PySide6 기반의 단일 창에서 세 단계(운동 선호도 조사 → 신체 측정 → 운동 자세 측정)가 페이지 전환 방식으로 진행됩니다.

## 주요 기능

- **신체 자세 측정**: 웹캠으로 정면/측면 사진을 촬영해 MediaPipe Pose로 어깨 기울기, 무릎 정렬(valgus), 거북목, 라운드 숄더, 흉추 후만, 골반 전방경사 각도를 계산합니다.
- **스트레칭 추천 & 운동 처방**: 정상 범위를 벗어난 항목을 기준으로 Ollama LLM이 스트레칭과 운동(스쿼트/런지/푸시업)을 추천합니다. 횟수·세트·중량은 안전 범위로 clamp됩니다.
- **실시간 운동 자세 판별**: 규칙 기반(각도 임계값) 판정으로 rep을 카운트하고, LSTM(TensorRT 엔진)으로 현재 운동 종류(pushup/squat/lunge/noaction)를 자동 인식합니다.

## 요구 사항

- 웹캠
- **NVIDIA GPU + CUDA**
- [Ollama](https://ollama.com/) 로컬 실행 및 `gemma3:4b` 모델

### Python 패키지

```bash
pip install PySide6 opencv-python numpy mediapipe ollama
pip install tensorrt pycuda   # NVIDIA GPU 환경에서 별도 설치
```

> `tensorrt` / `pycuda`는 CUDA 버전에 맞춰 설치해야 합니다. TensorRT 엔진(`.trt`)은 빌드된 환경의 GPU/드라이버에 종속되므로, 다른 장비에서는 `module/models/lstm_tanh_kfold.onnx`로부터 재빌드가 필요할 수 있습니다.

## 실행 방법

1. **Ollama 준비**

   ```bash
   ollama serve            # 백그라운드에서 Ollama 서버 실행
   ollama pull gemma3:4b   # 처방/추천에 사용하는 모델
   ```

2. **모델 파일 확인**

   아래 파일이 저장소에 포함되어 있어야 합니다.
   - `module/models/pose_landmarker_full.task` (MediaPipe Pose)
   - `module/models/lstm_tanh_kfold.trt` (운동 인식 TensorRT 엔진)

3. **애플리케이션 실행**

   프로젝트 루트에서 실행합니다. (모델 경로가 상대 경로로 지정되어 있어 루트에서 실행해야 합니다.)

   ```bash
   python main.py
   ```

### 조작 방법

- 신체 측정 화면에서 웹캠을 보며 GUI를 통해 촬영합니다. **정면 먼저, 이어서 측면** 순서로 두 장을 캡처합니다.
- 이후 LLM 추천/처방이 백그라운드에서 생성되고, 운동 자세 측정 단계로 넘어갑니다.

## 프로젝트 구조

```
main.py                     # GUI 진입점
gui/
  gui.py                    # 메인 UI (3단계 페이지, 카메라 스레드, LLM 워커)
  gui_style.py              # 스타일시트
module/
  mediapipe_op.py           # 포즈 추정, 이미지 캡처(CaptureForm), 랜드마크 추출
  cal_angle.py              # 3점 각도 계산
  upper_body.py / lower_body.py
  ollama_op.py              # LLM 스트레칭 추천 · 운동 처방 (gemma3:4b)
  workout_sel.py            # 처방된 운동 큐 관리
  workout_checker.py        # 규칙 기반 rep 카운트 + LSTM(TRT) 운동 인식
  basic_fn.py
  models/                   # MediaPipe .task, LSTM .onnx/.trt
```

## 참고

- 본 애플리케이션은 자세 교정 참고용이며 의료 진단을 제공하지 않습니다.
- `.trt` 엔진이 실행 환경과 호환되지 않으면 `.onnx` 모델로부터 TensorRT 엔진을 재생성해 사용하세요.
