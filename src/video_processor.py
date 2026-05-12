import cv2
import numpy as np
from PIL import Image

def load_video(video_path):
    """
    - open video file using cv2.VideoCapture
    - return cap, fps, total_frames, duration_sec
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps
    return cap, fps, total_frames, duration_sec

def extract_segment_frames(cap, fps, start_sec, end_sec, n_frames):
    """
    - start_sec ~ end_sec 구간에서 n_frames 개의 프레임을 균등 샘플링
   - np.linspace로 프레임 인덱스를 계산해라
   - 각 프레임을 BGR→RGB 변환 후 PIL Image로 만들어라
   - PIL Image 리스트를 반환해라
    :param cap:
    :param fps:
    :param start_sec:
    :param end_sec:
    :param n_frames:
    :return:
    """
    # 1) convert second to frames using fps
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    # 2) get random n numbers which are in (start_frame, end_frame)
    indices = np.linspace(start_frame, end_frame, n_frames, dtype=int)

    frames = []
    for idx in indices:

        # read idx-th frame
        # first, move the video to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # read frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert from BGR (OpenCV default) to RGB (MediaPipe expected format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #  PIL Image로 만들어 frames에 추가
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)

    return frames

def iter_segments(video_path, segment_sec, frames_per_seg):
    """
    - 영상 전체를 segment_sec 단위로 순회하는 제너레이터
    - 매 구간마다 (segment_id, start_sec, frames) 를 yield 해라
    - 영상 끝에서 구간이 남으면 버려라 (딱 떨어지는 구간만)
    :param video_path:
    :param segment_sec:
    :param frames_per_seg:
    :return:
    """
    cap, fps, total_frames, duration_sec = load_video(video_path)

    n_segments = int(duration_sec // segment_sec)  # 딱 떨어지는 구간 수

    try:
        for segment_id in range(n_segments):
            start_sec = segment_id * segment_sec
            end_sec = start_sec + segment_sec
            frames = extract_segment_frames(cap, fps, start_sec, end_sec, frames_per_seg)
            yield segment_id, start_sec, frames
    finally:
        cap.release()

# for seg_id, start_sec, frames in iter_segments("../data/duck.mp4", 5, 10):
#     print(seg_id, start_sec, len(frames))