"""
WetlandBirds -> Bimo Duck Agent 학습 라벨 어댑터

crops.csv (구간 단위) + bounding_boxes.csv (프레임 단위 박스)를 읽어서:
  1) 오리 3종(Northern Shoveler=5, Gadwall=11, Mallard=12)만 필터
  2) action_id를 3-class로 매핑 (feeding / resting / exploring), 나머지 drop
  3) 각 균질 구간(span)을 5초 타일링, 5초 미만 자투리는 버림  [방식 A]
  4) 각 5초 구간에서 10프레임 균등 샘플
  5) 출력:
     - labels_{video}.csv        : segment_id,start_sec,end_sec,label,notes  (기존 포맷 그대로)
     - duck_segment_boxes.csv    : video_name,segment_id,frame,x_min,y_min,x_max,y_max
                                   (video_processor가 crop 할 때 join key = video_name+segment_id)

주의:
- start_sec/end_sec는 "해당 원본 영상 내 절대 시각"이다 (duck1처럼 0부터가 아님).
- fps는 반드시 실제 영상 값으로 넣어야 5초 정의가 정확하다. 다운로드 후 ffprobe로 확인할 것.
  영상마다 fps가 다르면 --fps-map (JSON: {"043-mallard": 25, ...}) 사용.
- 박스 좌표는 min/max로 정렬해 파싱하므로 원본 컬럼 순서에 안전하다.
"""

import argparse
import ast
import csv
import json
import os
from collections import defaultdict

import numpy as np

DUCK_SPECIES = {5, 11, 12}          # Northern Shoveler, Gadwall, Mallard
LABEL_MAP = {0: "feeding", 6: "resting", 2: "exploring", 3: "exploring", 1: "resting"}
# drop: 1 Preening, 4 Alert, 5 Flying
SEG_SEC = 5
N_SAMPLE = 10


def load_boxes(path):
    """(video, frame) -> {bird_id: (x_min,y_min,x_max,y_max)}  (오리 구간 매칭용, 전 종 로드)"""
    boxes = defaultdict(dict)
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            video = row["video_name"]
            frame = int(row["frame"])
            raw = ast.literal_eval(row["bounding_boxes"])  # [(a,b,c,d, behavior_id, bird_id), ...]
            for t in raw:
                a, b, c, d, _behav, bird_id = t
                xs, ys = sorted((a, c)), sorted((b, d))
                boxes[(video, frame)][int(bird_id)] = (xs[0], ys[0], xs[1], ys[1])
    return boxes


def nearest_box(boxes, video, frame, bird_id, search=15):
    """정확 프레임에 해당 bird_id 박스가 없으면 ±search 내 가장 가까운 프레임에서 탐색."""
    for off in range(0, search + 1):
        for fr in (frame - off, frame + off):
            b = boxes.get((video, fr), {}).get(bird_id)
            if b is not None:
                return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="../data/wetlandbirds/crops.csv")
    ap.add_argument("--boxes", default="../data/wetlandbirds/bounding_boxes.csv")
    ap.add_argument("--out", default="../data/wetlandbirds")
    ap.add_argument("--fps", type=float, default=30.0, help="전 영상 공통 fps (기본 30)")
    ap.add_argument("--fps-map", default=None, help="영상별 fps JSON 경로 (없으면 --fps 사용)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fps_map = json.load(open(args.fps_map)) if args.fps_map else {}
    boxes = load_boxes(args.boxes)

    # video -> list of segment dicts
    per_video = defaultdict(list)
    class_counts = defaultdict(int)
    dropped_short = 0

    with open(args.crops, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        rows = list(r)

    # 안정적 segment_id를 위해 (video, start_frame, bird_id) 정렬
    rows = [row for row in rows if int(row["species_id"]) in DUCK_SPECIES
            and int(row["action_id"]) in LABEL_MAP]
    rows.sort(key=lambda x: (x["video_name"], int(x["start_frame"]), int(x["bird_id"])))

    box_rows = []  # companion

    for row in rows:
        video = row["video_name"]
        bird_id = int(row["bird_id"])
        label = LABEL_MAP[int(row["action_id"])]
        sf, ef = int(row["start_frame"]), int(row["end_frame"])
        fps = fps_map.get(video, args.fps)
        win = round(SEG_SEC * fps)  # 5초에 해당하는 프레임 수

        w = sf
        while w + win - 1 <= ef:                       # 방식 A: 완전한 5초 창만 사용
            w_end = w + win - 1
            sampled = np.unique(np.linspace(w, w_end, N_SAMPLE).round().astype(int))
            frame_boxes = []
            for fr in sampled:
                b = nearest_box(boxes, video, int(fr), bird_id)
                if b is not None:
                    frame_boxes.append((int(fr), b))
            # 박스가 하나도 없으면 그 구간은 crop 불가 -> skip
            if frame_boxes:
                seg = {
                    "start_sec": round(w / fps, 3),
                    "end_sec": round((w + win) / fps, 3),
                    "label": label,
                    "notes": f"wetlandbirds;bird{bird_id};frames{w}-{w_end}",
                    "frame_boxes": frame_boxes,
                }
                per_video[video].append(seg)
                class_counts[label] += 1
            w += win
        # 자투리(<5초)는 버림
        if (ef - sf + 1) % win:
            dropped_short += 1

    # 파일 출력
    for video, segs in per_video.items():
        segs.sort(key=lambda s: (s["start_sec"], s["notes"]))
        lpath = os.path.join(args.out, f"labels_{video}.csv")
        with open(lpath, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["segment_id", "start_sec", "end_sec", "label", "notes"])
            for i, s in enumerate(segs):
                wr.writerow([i, s["start_sec"], s["end_sec"], s["label"], s["notes"]])
                for fr, (xmn, ymn, xmx, ymx) in s["frame_boxes"]:
                    box_rows.append([video, i, fr,
                                     round(xmn, 2), round(ymn, 2), round(xmx, 2), round(ymx, 2)])

    with open(os.path.join(args.out, "duck_segment_boxes.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video_name", "segment_id", "frame", "x_min", "y_min", "x_max", "y_max"])
        wr.writerows(box_rows)

    # 요약
    total = sum(class_counts.values())
    print(f"fps={args.fps} (map={bool(fps_map)}) | duck videos={len(per_video)} | segments={total}")
    for k in ("feeding", "resting", "exploring"):
        c = class_counts[k]
        print(f"  {k:10} {c:4}  ({100*c/total:.1f}%)" if total else f"  {k:10} 0")
    print(f"  spans with discarded remainder: {dropped_short}")
    print(f"  box rows: {len(box_rows)}")


if __name__ == "__main__":
    main()