import clip
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from behaviour_states import BehaviourClass, DEFAULT_CONFIDENCE_THRESHOLD
from video_processor import extract_segment_frames, load_video, iter_segments

# ──────────────────────────────────────────────
# CLIP 모델 캐시
# ──────────────────────────────────────────────
_clip_cache = {}

def get_clip_model(model_name):
    """
    CLIP 모델을 전역변수로 한 번만 로드 (재호출 시 캐시 반환)
    반환: model, preprocess, device (실제 torch.device 객체)
    """
    global _clip_cache
    if model_name in _clip_cache:
        return _clip_cache[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(model_name, device=device)
    clip_model.eval()

    _clip_cache[model_name] = (clip_model, preprocess, device)
    print(f"[{datetime.now():%H:%M:%S}] [classifier] CLIP 로드 완료: {model_name} on {device}")
    return clip_model, preprocess, device


# ──────────────────────────────────────────────
# 임베딩 추출 (배치 처리)
# ──────────────────────────────────────────────
def frames_to_embedding(frames, model_name, batch_size=32):
    """
    PIL Image 리스트 → CLIP 배치 추출 → L2 정규화 → 평균 벡터 반환 (shape: 512,)

    변경점:
    - 기존: 프레임 하나씩 encode_image() 호출
    - 개선: batch_size 단위로 묶어서 한 번에 처리 → 3~5배 빠름
    - 버그 수정: device를 실제 torch.device 객체로 사용
    """
    model, preprocess, device = get_clip_model(model_name)

    # 전체 프레임을 텐서로 변환
    tensors = torch.stack([preprocess(f) for f in frames]).to(device)  # (N, 3, 224, 224)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size]
            emb = model.encode_image(batch)                              # (B, 512)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            embeddings.append(emb)

    stacked = torch.cat(embeddings, dim=0)           # (N, 512)
    mean_emb = stacked.mean(dim=0)                   # (512,)
    mean_emb = torch.nn.functional.normalize(mean_emb, dim=-1)
    return mean_emb.cpu().numpy()


# ──────────────────────────────────────────────
# 피처 매트릭스 빌드 (캐시 지원)
# ──────────────────────────────────────────────
def build_feature_matrix(video_path, labels_csv, model_name, cache_path=None):
    """
    labels.csv → 프레임 추출 → CLIP 임베딩 → X, y 반환

    변경점:
    - cache_path 지정 시 첫 실행에 .npz 저장, 이후 실행은 캐시 로드 (2.5h → 수초)
    - uncertain 라벨 제외
    """
    # ── 캐시 확인 ──
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            print(f"[{datetime.now():%H:%M:%S}] [classifier] 캐시 로드: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            X, y = data["X"], data["y"]
            print(f"[{datetime.now():%H:%M:%S}] [classifier] 캐시에서 X={X.shape}, y={y.shape} 로드 완료")
            return X, y

    # ── CSV 로드 ──
    label_df = pd.read_csv(labels_csv, index_col=False)
    label_df = label_df[label_df["label"] != "uncertain"].reset_index(drop=True)

    if label_df.empty:
        raise ValueError("uncertain 제외 후 유효한 라벨 행이 없습니다.")

    cap, fps, total_frames, duration_sec = load_video(video_path)
    print(f"[{datetime.now():%H:%M:%S}] fps: {fps}, total_frames: {total_frames}, duration_sec: {duration_sec}")

    X, y = [], []
    total = len(label_df)

    for idx, (_, row) in enumerate(label_df.iterrows()):
        start = int(row["start_sec"])
        end   = int(row["end_sec"])
        label = str(row["label"])

        # 진행률 출력 (10구간마다)
        if idx % 10 == 0:
            print(f"[{datetime.now():%H:%M:%S}] [classifier] 진행: {idx}/{total} 구간")

        try:
            frames = extract_segment_frames(cap, fps, start, end, (end - start) * fps)
            if not frames:
                print(f"[{datetime.now():%H:%M:%S}] [classifier] 프레임 없음 → {start}~{end}s 건너뜀")
                continue

            embedding = frames_to_embedding(frames, model_name=model_name)
            X.append(embedding)
            y.append(label)

        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] [classifier] 구간 {start}~{end}s 처리 실패: {e}")
            continue

    if not X:
        raise RuntimeError("임베딩 추출에 성공한 구간이 없습니다.")

    X = np.stack(X)   # (N, 512)
    y = np.array(y)   # (N,)

    print(f"[{datetime.now():%H:%M:%S}] [classifier] 피처 매트릭스 완성: X={X.shape}, y={y.shape}")
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 클래스 분포: { {c: int((y == c).sum()) for c in np.unique(y)} }")

    # ── 캐시 저장 ──
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, y=y)
        print(f"[{datetime.now():%H:%M:%S}] [classifier] 캐시 저장 완료: {cache_path}")

    return X, y


# ──────────────────────────────────────────────
# 분류기 학습
# ──────────────────────────────────────────────
def train_classifier(X, y, model_save_path, test_size=0.2, random_state=42):
    """
    LabelEncoder → train_test_split → LogisticRegression → 평가 → joblib 저장
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"[{datetime.now():%H:%M:%S}] [classifier] 클래스 매핑: "
          f"{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n[{datetime.now():%H:%M:%S}] [classifier] === Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_,
                                zero_division=0))   # UndefinedMetricWarning 제거

    payload = {"clf": clf, "le": label_encoder}
    joblib.dump(payload, model_save_path)
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 모델 저장 완료: {model_save_path}")
    return clf, label_encoder


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_classifier(model_path):
    """joblib으로 모델 불러와서 clf, le 반환"""
    payload = joblib.load(model_path)
    clf, le = payload["clf"], payload["le"]
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 모델 로드 완료: {model_path}")
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 클래스: {list(le.classes_)}")
    return clf, le


# ──────────────────────────────────────────────
# 세그먼트 예측
# ──────────────────────────────────────────────
def predict_segment(frames, clf, le, confidence_threshold, clip_model):
    """
    프레임 리스트 → 임베딩 → predict_proba → (BehaviourClass, confidence) 반환
    최대 확률 < confidence_threshold 이면 UNCERTAIN 반환
    """
    if not frames:
        print(f"[{datetime.now():%H:%M:%S}] [classifier] 빈 프레임 → UNCERTAIN 반환")
        return BehaviourClass.UNCERTAIN, 0.0

    try:
        embedding = frames_to_embedding(frames, model_name=clip_model)
        proba = clf.predict_proba(embedding.reshape(1, -1))[0]   # (n_classes,)
        max_confidence = float(proba.max())
        predicted_idx  = int(proba.argmax())

        if max_confidence < confidence_threshold:
            return BehaviourClass.UNCERTAIN, max_confidence

        label_str = le.inverse_transform([predicted_idx])[0]
        behaviour = BehaviourClass(label_str)
        return behaviour, max_confidence

    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] [classifier] predict_segment 오류: {e}")
        return BehaviourClass.UNCERTAIN, 0.0