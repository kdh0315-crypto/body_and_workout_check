import numpy as np



def array_to_pixel(point, width, height):
    """정규화 좌표 [x, y]를 OpenCV 픽셀 좌표로 변환한다."""
    return int(point[0] * width), int(point[1] * height)

def midpoint(point_a, point_b):
    """MediaPipe Landmark 두 점의 정규화된 2차원 중간점을 계산한다."""
    return np.array(
        [
            (point_a.x + point_b.x) / 2.0,
            (point_a.y + point_b.y) / 2.0,
        ],
        dtype=np.float32,
    )