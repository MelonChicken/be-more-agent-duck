import cv2
import csv
import numpy as np
import os
from collections import defaultdict
from PIL import Image

from src.detector import detect_bird_box

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
    duration_sec = int(total_frames / fps)
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
    n_frames = int(n_frames)
    # print(f"start_frame={start_frame}, end_frame={end_frame}, n_frames={n_frames}")
    # 2) get random n numbers which are in (start_frame, end_frame)
    indices = np.linspace(start_frame, end_frame, n_frames).astype(int).tolist()

    frames = []
    for idx in indices:
        # print(f"current idx : {idx}")
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

def _read_segment_boxes(boxes_csv):
    """duck_segment_boxes.csv -> {(video_name, segment_id): [(frame,x_min,y_min,x_max,y_max),...]}"""
    table = defaultdict(list)
    with open(boxes_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["video_name"], int(row["segment_id"]))
            table[key].append((int(row["frame"]),
                               float(row["x_min"]), float(row["y_min"]),
                               float(row["x_max"]), float(row["y_max"])))
    for k in table:
        table[k].sort(key=lambda r: r[0])
    return table

def _crop_frame(cap, frame_idx, box, pad=0.10, min_size=32):
    """frame_idx를 box로 crop -> PIL RGB. 실패/과소 시 None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = box
    bw, bh = x_max - x_min, y_max - y_min
    x_min -= bw * pad; x_max += bw * pad
    y_min -= bh * pad; y_max += bh * pad
    x_min = max(0, int(x_min)); y_min = max(0, int(y_min))
    x_max = min(w, int(x_max)); y_max = min(h, int(y_max))
    if x_max - x_min < min_size or y_max - y_min < min_size:
        return None
    crop = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)

def _crop_frame_arr(frame_bgr, box, pad=0.10, min_size=32):
    """Already-read BGR frame -> box crop -> PIL RGB. Returns None if invalid/tiny."""
    h, w = frame_bgr.shape[:2]
    x_min, y_min, x_max, y_max = box
    bw, bh = x_max - x_min, y_max - y_min
    x_min -= bw * pad; x_max += bw * pad
    y_min -= bh * pad; y_max += bh * pad
    x_min = max(0, int(x_min)); y_min = max(0, int(y_min))
    x_max = min(w, int(x_max)); y_max = min(h, int(y_max))
    if x_max - x_min < min_size or y_max - y_min < min_size:
        return None
    crop = cv2.cvtColor(frame_bgr[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)

def iter_segments_from_boxes(video_path, video_name, boxes_table):
    """학습셋용: companion 프레임/박스로 crop. iter_segments와 동일 시그니처.
       (segment_id, start_sec, frames[PIL]) yield."""
    cap, fps, _, _ = load_video(video_path)
    seg_ids = sorted({sid for (vn, sid) in boxes_table if vn == video_name})
    try:
        for segment_id in seg_ids:
            rows = boxes_table[(video_name, segment_id)]
            frames = []
            for frame_idx, *box in rows:
                pil = _crop_frame(cap, frame_idx, box)
                if pil is not None:
                    frames.append(pil)
            if not frames:
                continue
            start_sec = rows[0][0] / fps
            yield segment_id, start_sec, frames
    finally:
        cap.release()

def iter_segments_detected(video_path, segment_sec, frames_per_seg,
                           detector, conf_threshold=0.25, imgsz=640,
                           pad=0.10, min_size=32, fallback="prev"):
    """
    Evaluation/inference path: full-frame video -> detect(bird) -> crop.

    Yields the same shape as iter_segments: (segment_id, start_sec, frames[PIL]).
    fallback:
      - "prev": reuse the previous valid box; if unavailable, use full-frame
      - "full": use the current full-frame
      - "skip": drop the frame
    """
    if fallback not in {"prev", "full", "skip"}:
        raise ValueError("fallback must be one of: 'prev', 'full', 'skip'")

    cap, fps, total_frames, duration_sec = load_video(video_path)
    n_segments = int(duration_sec // segment_sec)
    prev_box = None

    try:
        for segment_id in range(n_segments):
            start_sec = segment_id * segment_sec
            end_sec = start_sec + segment_sec
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            indices = np.linspace(start_frame, end_frame, int(frames_per_seg)).astype(int).tolist()

            frames = []
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                box = detect_bird_box(
                    detector,
                    frame_pil,
                    conf_threshold=conf_threshold,
                    imgsz=imgsz,
                )

                if box is not None:
                    crop = _crop_frame_arr(frame, box, pad=pad, min_size=min_size)
                    if crop is not None:
                        crop.info["detect_source"] = "detected"
                        frames.append(crop)
                        prev_box = box
                        continue

                if fallback == "prev" and prev_box is not None:
                    crop = _crop_frame_arr(frame, prev_box, pad=pad, min_size=min_size)
                    if crop is not None:
                        crop.info["detect_source"] = "fallback_prev"
                        frames.append(crop)
                        continue

                if fallback == "full" or (fallback == "prev" and prev_box is None):
                    frame_pil.info["detect_source"] = "fallback_full"
                    frames.append(frame_pil)

            if frames:
                detected_count = sum(f.info.get("detect_source") == "detected" for f in frames)
                prev_count = sum(f.info.get("detect_source") == "fallback_prev" for f in frames)
                full_count = sum(f.info.get("detect_source") == "fallback_full" for f in frames)
                if prev_count or full_count:
                    print(
                        f"[video_processor] segment {segment_id}: "
                        f"detected={detected_count}, "
                        f"fallback_prev={prev_count}, "
                        f"fallback_full={full_count}"
                    )
                yield segment_id, start_sec, frames
    finally:
        cap.release()
