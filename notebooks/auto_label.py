# notebooks/auto_label.py
#
# 역할: 오리 영상을 5초 구간으로 쪼개고, 각 구간의 프레임들을 LLaVA(로컬 VLM)에게
#       보내 행동 라벨을 자동으로 붙인다. 결과는 data/labels_{영상이름}.csv 로 저장되며
#       이후 classifier.py 학습에 사용된다.
#
# ✅ v2 변경점 (실험 #5 진단 반영)
#   1) 라벨-피처 창(window) 통일: 학습과 "똑같은" 프레임을 쓴다.
#      기존은 구간당 1프레임(1.5초 지점)만 보고 라벨을 정했지만, classifier는 구간의
#      10프레임 평균으로 임베딩을 만든다 → 라벨과 피처가 서로 다른 순간을 가리켰다.
#      이제 src.video_processor.extract_segment_frames 를 재사용해 학습과 동일한
#      프레임을 뽑고, 그중 여러 장을 다수결로 라벨링한다.
#   2) 다수결 + 불일치→uncertain: feeding처럼 간헐적인 행동을 1프레임 운으로 잘못
#      찍는 문제를 완화. 프레임 간 응답이 갈리면 uncertain으로 빼서 학습셋을 깨끗하게.
#      (다수결 결과는 "구간을 평균으로 요약한 임베딩"과 같은 방향을 가리킨다)
#   3) 파싱 보강: 정확 매칭(첫 토큰 → whole-word) 우선, 실패 시에만 substring 폴백.
#      키워드가 여러 개 섞이면 모호 처리(무효표).
#
# 실행 위치: be-more-agent-duck/notebooks/ 에서 실행
# 실행 방법: python auto_label.py

import sys, os
sys.path.insert(0, '..')   # 루트 경로 추가 (config.json, src 접근용)

import re
import csv
import json
import base64
from io import BytesIO
from collections import Counter
from PIL import Image
import ollama

# 학습과 동일한 프레임 추출기를 그대로 재사용한다 (통일의 핵심)
from src.video_processor import load_video, extract_segment_frames


# ── 설정 로드 ────────────────────────────────────────────────────────────────
with open(os.path.join('..', 'config.json')) as f:
    cfg = json.load(f)

VIDEO_PATH     = os.path.join('..', cfg['video_path'])   # 분석할 영상 파일 경로
SEGMENT_SEC    = cfg['segment_sec']                      # 구간 길이 (기본 5초)
FRAMES_PER_SEG = cfg['frames_per_seg']                   # 구간당 프레임 수 (기본 10) — 학습과 동일
LABEL_MODEL    = cfg.get('label_model', 'llava:7b')      # 라벨링에 쓸 ollama 모델

# 출력 파일명을 영상 stem 기준으로 생성 → classifier.load_or_train_classifier가
# 기대하는 labels_{stem}.csv 규칙과 일치시킨다.
_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_CSV = os.path.join('..', 'data','train', f'labels_{_stem}.csv')

# 다수결 설정
VOTE_FRAMES     = cfg.get('label_vote_frames', 3)     # 구간에서 몇 장을 투표에 쓸지
CONSENSUS_RATIO = cfg.get('label_consensus', 0.6)     # 다수표 비율이 이 값 미만이면 uncertain
# ─────────────────────────────────────────────────────────────────────────────


LABELS = ['resting', 'feeding', 'exploring', 'uncertain']

# 프레임 1장을 독립적으로 질의한다 (뒤에서 여러 장을 모아 다수결).
PROMPT = """이 이미지 속 오리의 행동을 아래 4가지 중 하나로만 판단하세요.
판단 기준은 '몸이 움직이는가'가 아니라 '머리와 부리가 어디를 향하는가'입니다.
다른 말은 절대 하지 말고, 한 단어만 출력하세요.

- feeding  : 머리·목을 아래로 숙여 부리가 바닥/물/풀에 닿아 먹이를 쪼거나 훑음,
             또는 물속에 머리를 넣거나 꼬리를 들고 물에 거꾸로 선 상태.
             ★ 몸통이 멈춰 있어도 부리가 아래 먹이를 향하면 feeding입니다.
- resting  : 부리를 등·날개 깃에 파묻고 있거나, 머리를 세운 채 눈을 감고 가만히 있음.
             머리가 아래 먹이를 향하지 않습니다.
- exploring: 머리를 들고 주위를 살피며 걷거나 헤엄쳐 위치가 이동함.
- uncertain: 오리가 잘 안 보이거나 위 셋으로 판단이 어려움
             (예: 부리로 자기 깃털을 다듬는 중이면 uncertain).

한 단어만 출력:"""


def frame_to_base64(pil_img: Image.Image) -> str:
    """PIL Image → base64(JPEG) 문자열. ollama는 이미지를 base64로 받는다."""
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask_llava_raw(pil_img: Image.Image) -> str:
    """프레임 1장을 LLaVA에 보내 원문 응답(문자열)을 그대로 반환. 파싱은 분리."""
    img_b64 = frame_to_base64(pil_img)
    response = ollama.chat(
        model=LABEL_MODEL,
        messages=[{'role': 'user', 'content': PROMPT, 'images': [img_b64]}],
    )
    return response['message']['content']


def parse_label(raw: str) -> str | None:
    """
    LLaVA 원문 응답 → 라벨 1개 또는 None.

    기존 파싱의 취약점:
      for label in LABELS: if label in raw  →  "resting이 아니라 feeding"처럼
      키워드가 여러 개 들어오면 리스트 순서상 앞의 것(resting)을 잘못 반환했다.

    보강 순서:
      1) 첫 줄의 첫 단어가 라벨이면 그것 (모델은 보통 정답을 먼저 말한다)
      2) 전체에서 whole-word로 매칭된 '서로 다른' 라벨이 정확히 1개면 그것
         (2개 이상 섞이면 모호 → None)
      3) 위가 다 실패하면 substring 폴백, 역시 distinct 1개일 때만
    반환:
      'resting'|'feeding'|'exploring'|'uncertain' 또는 None(파싱 불가 → 상위에서 무효표)
    """
    text = raw.strip().lower()
    if not text:
        return None

    # 1) 첫 줄 첫 토큰
    first_line = text.split('\n', 1)[0]
    for tok in re.findall(r'[a-z]+', first_line):
        if tok in LABELS:
            return tok

    # 2) whole-word 매칭 (distinct 1개만 인정)
    found = {lab for lab in LABELS if re.search(rf'\b{lab}\b', text)}
    if len(found) == 1:
        return next(iter(found))
    if len(found) > 1:
        return None   # 여러 라벨 혼재 → 모호

    # 3) substring 폴백 (distinct 1개만 인정)
    found_sub = {lab for lab in LABELS if lab in text}
    if len(found_sub) == 1:
        return next(iter(found_sub))
    return None


def pick_vote_indices(n_frames: int, k: int) -> list[int]:
    """n_frames개 중 균등 간격으로 k개 인덱스 선택 (양끝 포함)."""
    if k >= n_frames:
        return list(range(n_frames))
    if k <= 1:
        return [n_frames // 2]
    return [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]


def label_segment(frames: list) -> tuple[str, Counter]:
    """
    구간 프레임들(학습과 동일) → 다수결 라벨 + 표 집계.

    - VOTE_FRAMES 장을 균등 추출해 각각 독립 질의
    - 파싱 실패/모호(None)와 'uncertain' 응답은 모두 uncertain 표로 계산
    - 최다표가 실제 행동이고 그 비율이 CONSENSUS_RATIO 이상이면 그 라벨,
      아니면(불일치가 크면) uncertain → 학습셋에서 자연스럽게 걸러진다
    """
    if not frames:
        return 'uncertain', Counter()

    idxs = pick_vote_indices(len(frames), VOTE_FRAMES)
    votes = Counter()
    for i in idxs:
        parsed = parse_label(ask_llava_raw(frames[i]))
        votes[parsed if parsed is not None else 'uncertain'] += 1

    top_label, top_n = votes.most_common(1)[0]
    if top_label != 'uncertain' and (top_n / len(idxs)) >= CONSENSUS_RATIO:
        return top_label, votes
    return 'uncertain', votes


def votes_to_notes(votes: Counter) -> str:
    """표 집계를 CSV notes 칸에 남겨 수동 검토를 돕는다. 예: 'feeding:2;resting:1'"""
    return ';'.join(f'{k}:{v}' for k, v in sorted(votes.items(), key=lambda x: -x[1]))


def main():
    """
    전체 라벨링 파이프라인.
      1) load_video로 영상 정보 파악 (cap을 한 번만 연다 → 구간마다 재오픈 안 함)
      2) 구간마다 학습과 동일한 프레임 추출 → 다수결 라벨 → CSV 기록
      3) data/labels_{stem}.csv 저장
    """
    cap, fps, total_frames, duration_sec = load_video(VIDEO_PATH)
    total_segments = int(duration_sec // SEGMENT_SEC)

    print(f"영상: {VIDEO_PATH}")
    print(f"길이 {duration_sec:.1f}s → {total_segments}구간, 구간당 {VOTE_FRAMES}표 질의")
    print(f"모델 {LABEL_MODEL} | 합의 임계 {CONSENSUS_RATIO} | 출력 {OUTPUT_CSV}\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    dist = Counter()
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 헤더: classifier.build_feature_matrix()가 이 컬럼명을 기대함
        writer.writerow(['segment_id', 'start_sec', 'end_sec', 'label', 'notes'])

        for seg_id in range(total_segments):
            start_sec = seg_id * SEGMENT_SEC
            end_sec   = start_sec + SEGMENT_SEC

            # ★ 학습과 동일한 프레임 추출 (통일 지점)
            frames = extract_segment_frames(cap, fps, start_sec, end_sec, FRAMES_PER_SEG)

            if not frames:
                label, votes = 'uncertain', Counter()
            else:
                label, votes = label_segment(frames)

            dist[label] += 1
            writer.writerow([seg_id, start_sec, end_sec, label, votes_to_notes(votes)])
            print(f"  [{seg_id:3d}] {start_sec:5.0f}s~{end_sec:5.0f}s → {label:9s} ({votes_to_notes(votes)})")

    cap.release()

    print(f"\n✅ 완료 → {OUTPUT_CSV}")
    print(f"클래스 분포: {dict(dist)}")
    print("feeding 구간은 별도로 재생해 검증 권장 (소수 클래스는 precision이 중요).")


if __name__ == '__main__':
    main()