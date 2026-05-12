# notebooks/auto_label.py
#
# 역할: 오리 영상을 5초 구간으로 쪼개고, 각 구간의 대표 프레임을
#       LLaVA(로컬 VLM)에게 보내서 행동 라벨을 자동으로 붙여주는 스크립트.
#       결과는 data/labels.csv 로 저장되며, 이후 classifier.py 학습에 사용된다.
#
# 실행 위치: be-more-agent-duck/notebooks/ 에서 실행
# 실행 방법: python auto_label.py

import sys, os
sys.path.insert(0, '..')   # 루트 경로 추가 (config.json 접근용)

import cv2
import csv
import base64
import json
import time
from io import BytesIO
from PIL import Image
import ollama


# ── 설정 로드 ────────────────────────────────────────────────────────────────
# config.json 에서 영상 경로, 구간 길이, 프레임 수 등을 읽어온다.
# 이 값들을 여기서 하드코딩하지 않고 config에서 읽는 이유는
# 나중에 파라미터를 바꿀 때 config.json 한 곳만 수정하면 되기 때문.
with open('../config.json') as f:
    cfg = json.load(f)

VIDEO_PATH     = os.path.join('..', cfg['video_path'])   # 분석할 영상 파일 경로
OUTPUT_CSV     = '../data/labels.csv'                    # 라벨 결과 저장 경로
SEGMENT_SEC    = cfg['segment_sec']                      # 구간 길이 (기본 5초)
FRAMES_PER_SEG = cfg['frames_per_seg']                   # 구간당 프레임 수 (기본 10)
SAMPLE_FRAME   = 3   # 구간 내 몇 번째 프레임을 VLM에 보낼지 (0~FRAMES_PER_SEG-1)
                     # 0이면 구간 시작, 4~5면 구간 중간 부근
# ─────────────────────────────────────────────────────────────────────────────


# LLaVA에게 보낼 프롬프트.
# "한 단어만 출력"을 강조하는 이유: LLaVA가 문장으로 길게 답하면
# 파싱이 어려워지기 때문에 최대한 짧게 유도한다.
PROMPT = """이 이미지에 오리가 있습니다.
오리의 행동을 아래 4가지 중 하나로만 답하세요. 다른 말은 절대 하지 마세요.

- resting  : 가만히 있거나 졸고 있음
- feeding  : 먹이를 먹거나 물에 머리를 넣음
- exploring: 걷거나 헤엄치며 이동/탐색
- uncertain: 오리가 잘 안 보이거나 판단 불가

한 단어만 출력:"""


def frame_to_base64(pil_img: Image.Image) -> str:
    """
    PIL Image → base64 문자열 변환.

    ollama API는 이미지를 파일 경로가 아닌 base64 인코딩된 문자열로 받는다.
    JPEG 품질 85로 압축해서 전송 크기를 줄이고,
    BytesIO를 사용해 디스크에 임시 파일을 쓰지 않고 메모리에서 처리한다.

    Args:
        pil_img: 변환할 PIL Image 객체

    Returns:
        base64로 인코딩된 JPEG 이미지 문자열
    """
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def extract_frame(video_path: str, sec: float) -> Image.Image | None:
    """
    영상에서 특정 시각(초)의 프레임 한 장을 추출해 PIL Image로 반환.

    OpenCV의 CAP_PROP_POS_FRAMES으로 원하는 프레임 위치로 이동한 뒤
    한 장만 읽고 바로 cap을 해제한다. (매번 열고 닫는 이유: 구간마다
    독립적으로 접근하기 때문에 전체를 열어두는 것보다 메모리가 안전함)

    OpenCV는 BGR 포맷으로 읽기 때문에 PIL에서 쓰려면 RGB로 변환 필요.

    Args:
        video_path: 영상 파일 경로
        sec: 추출할 시각 (초 단위, float)

    Returns:
        RGB PIL Image. 읽기 실패 시 None 반환.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))  # 원하는 위치로 이동
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # BGR(OpenCV 기본) → RGB(PIL 기본) 변환
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def ask_llava(pil_img: Image.Image) -> str:
    """
    PIL Image를 LLaVA에게 보내서 행동 라벨(문자열)을 받아온다.

    LLaVA가 항상 정확히 한 단어만 반환하지는 않기 때문에,
    응답 텍스트 안에 라벨 키워드가 포함되어 있는지 순서대로 확인한다.
    아무 라벨도 파싱되지 않으면 'uncertain'으로 처리한다.

    Args:
        pil_img: 분석할 프레임 이미지

    Returns:
        'resting' | 'feeding' | 'exploring' | 'uncertain' 중 하나
    """
    img_b64 = frame_to_base64(pil_img)

    response = ollama.chat(
        model='llava:7b',
        messages=[{
            'role': 'user',
            'content': PROMPT,
            'images': [img_b64]
        }]
    )

    raw = response['message']['content'].strip().lower()

    # 응답에서 알려진 라벨 키워드 탐색
    # 순서가 중요: 더 구체적인 것을 앞에 둠
    for label in ['resting', 'feeding', 'exploring', 'uncertain']:
        if label in raw:
            return label

    # 어떤 키워드도 매칭 안 되면 uncertain 처리
    return 'uncertain'


def main():
    """
    전체 라벨링 파이프라인 실행.

    1. 영상 전체 길이를 파악해서 총 구간 수 계산
    2. 구간마다 대표 프레임 추출 → LLaVA 질의 → 결과 CSV에 기록
    3. 완료 후 data/labels.csv 저장

    구간 수가 많을 경우 시간이 걸릴 수 있다.
    (LLaVA 7B 기준 구간당 약 3~10초, 60구간이면 3~10분 소요)
    """
    # 영상 기본 정보 파악
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    total_segments = int(duration_sec // SEGMENT_SEC)
    print(f"영상 길이: {duration_sec:.1f}초 → {total_segments}구간 처리 예정")
    print(f"예상 소요 시간: {total_segments * 5 // 60}분 ~ {total_segments * 10 // 60}분\n")

    os.makedirs('../data', exist_ok=True)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # CSV 헤더: classifier.py의 build_feature_matrix()가 이 컬럼명을 기대함
        writer.writerow(['segment_id', 'start_sec', 'end_sec', 'label', 'notes'])

        for seg_id in range(total_segments):
            start_sec = seg_id * SEGMENT_SEC
            end_sec   = start_sec + SEGMENT_SEC

            # 구간 내 SAMPLE_FRAME 번째 프레임의 실제 시각 계산
            # ex) SEGMENT_SEC=5, FRAMES_PER_SEG=10, SAMPLE_FRAME=3
            #     → 구간 시작 + 1.5초 지점 프레임
            sample_sec = start_sec + (SEGMENT_SEC / FRAMES_PER_SEG) * SAMPLE_FRAME

            img = extract_frame(VIDEO_PATH, sample_sec)

            if img is None:
                # 프레임 추출 실패 (영상 끝 부분 등)
                label = 'uncertain'
            else:
                label = ask_llava(img)

            writer.writerow([seg_id, start_sec, end_sec, label, ''])
            print(f"  [{seg_id:3d}] {start_sec:5.0f}s ~ {end_sec:5.0f}s  →  {label}")

            # ollama에 요청이 몰리지 않도록 짧은 딜레이
            time.sleep(0.3)

    print(f"\n✅ 완료 → {OUTPUT_CSV}")
    print("이상한 라벨은 CSV 열어서 수동으로 수정하면 됩니다.")


if __name__ == '__main__':
    main()
