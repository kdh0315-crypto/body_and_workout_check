import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def process_image(image_path, output_dir, mode="skeleton_only", save_json=True):
    """
    한 장의 이미지를 처리해 관절이 표시된 이미지를 저장.
    mode: "overlay"       -> 원본 위에 스켈레톤
          "skeleton_only" -> 검은 배경에 관절만
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIP] 읽기 실패: {image_path}")
        return None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,      # 단일 이미지는 True
        model_complexity=2,          # 0/1/2, 2가 가장 정확(느림)
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(rgb)

    if not results.pose_landmarks:
        print(f"[NO POSE] 관절 미검출: {image_path}")
        return None

    # 출력 캔버스 선택
    if mode == "overlay":
        canvas = image.copy()
    else:  # skeleton_only
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 관절 + 연결선 그리기
    mp_drawing.draw_landmarks(
        canvas,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=2),
    )

    # 저장
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_pose.png")
    cv2.imwrite(out_path, canvas)

    # 좌표 JSON 저장 (자세 분석용)
    if save_json:
        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "id": idx,
                "name": mp_pose.PoseLandmark(idx).name,
                "x_px": lm.x * w,
                "y_px": lm.y * h,
                "z": lm.z,              # 상대 깊이
                "visibility": lm.visibility,
            })
        json_path = os.path.join(output_dir, f"{base}_landmarks.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(landmarks, f, ensure_ascii=False, indent=2)

    print(f"[OK] {out_path}")
    return out_path


def batch_process(input_dir, output_dir, mode="skeleton_only"):
    os.makedirs(output_dir, exist_ok=True)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
        files.extend(glob.glob(os.path.join(input_dir, e.upper())))

    print(f"총 {len(files)}개 이미지 처리 시작")
    for path in sorted(files):
        process_image(path, output_dir, mode=mode)


if __name__ == "__main__":
    batch_process(
        input_dir="./input_images",
        output_dir="./output_images",
        mode="skeleton_only",   # 또는 "overlay"
    )