import clip
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from behaviour_states import BehaviourClass, DEFAULT_CONFIDENCE_THRESHOLD
from video_processor import extract_segment_frames, load_video, iter_segments

_clip_cache = {}
def get_clip_model(model_name):
    """
    - CLIP 모델을 전역변수로 한 번만 로드해라 (두 번 호출해도 다시 로드 안 함)
    - cuda 사용 가능하면 cuda, 아니면 cpu 사용
    - model, preprocess, device 반환
    """
    global _clip_cache
    # if there is already model, return that model, preprocess, device
    if model_name in _clip_cache:
        return _clip_cache[model_name]

    clip_model, preprocess = clip.load(model_name)

    if torch.cuda.is_available():
        torch.device('cuda')
        clip_model = clip_model.cuda()
    else:
        torch.device('cpu')
        clip_model = clip_model.cpu()
    _clip_cache[model_name] = (clip_model, preprocess, torch.device)

    return clip_model, preprocess, torch.device


def frames_to_embedding(frames, model_name):
    """
    - PIL Image 리스트를 받아서
    - 각 프레임을 CLIP으로 임베딩 추출
    - L2 정규화 후 프레임들의 평균 벡터 반환 (shape: 512,)
    - torch.no_grad() 안에서 실행해라
    """
    # Pytorch doesn't track the gradient change which may be used in back propagation
    model, preprocess, device = get_clip_model(model_name)
    embeddings = []

    with torch.no_grad():
        for frame in frames:
            image_tensor = preprocess(frame).unsqueeze(0) # convert PIL image to tensor, and add batch information, add device information
            embedding = model.encode_image(image_tensor)
            embedding = torch.nn.functional.normalize(embedding, dim=-1) # 마지막 차원 기준 정규화 (dim=-1)
            embeddings.append(embedding)

    stacked = torch.cat(embeddings, dim=0)   # [10, 512]
    mean_embedding = stacked.mean(dim=0)     # [512]
    mean_embedding = torch.nn.functional.normalize(mean_embedding, dim=-1) # 재 정규화
    return mean_embedding.cpu().numpy()


def build_feature_matrix(video_path, labels_csv, model_name):
    """
    - labels.csv를 pandas로 읽어라
    - label == "uncertain" 인 행은 제외해라
    - 각 구간의 프레임을 추출하고 임베딩을 뽑아서
    - X (numpy array), y (label 문자열 array) 반환
    """
    label_df = pd.read_csv(labels_csv, index_col=False)
    label_df = label_df[label_df["label"] != "uncertain"].reset_index(drop=True)

    if label_df.empty:
        raise ValueError("uncertain 제외 후 유효한 라벨 행이 없습니다.")

    cap, fps, total_frames, duration_sec = load_video(video_path)
    print(f"[{datetime.now():%H:%M:%S}] fps: {fps}, total_frames: {total_frames}, duration_sec: {duration_sec}")

    X = []
    y = []

    for _, row in label_df.iterrows():
        start = int(row["start_sec"])
        end = int(row["end_sec"])
        label = str(row["label"])

        # print(f"start_sec={start}, end_sec={end}, fps={fps}")
        try:
            frames = extract_segment_frames(cap, fps, start, end, (end-start)*fps)
            if not frames:
                print(f"[{datetime.now():%H:%M:%S}] [classifier] 프레임 없음 → 구간 {start:.1f}~{end:.1f}s 건너뜀")
                continue

            embedding = frames_to_embedding(frames, model_name=model_name)
            X.append(embedding)
            y.append(label)

        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] [classifier] 구간 {start}~{end}s 처리 실패: {e}")
            continue

    if not X:
        raise RuntimeError("임베딩 추출에 성공한 구간이 없습니다.")

    X = np.stack(X)  # (N, 512)
    y = np.array(y)  # (N,)

    print(f"[{datetime.now():%H:%M:%S}] [classifier] 피처 매트릭스 완성: X={X.shape}, y={y.shape}")
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 클래스 분포: { {c: int((y == c).sum()) for c in np.unique(y)} }")

    return X, y

def train_classifier(X, y, model_save_path, test_size=0.2, random_state=42):
   """
   - LabelEncoder로 y를 숫자로 변환
   - train_test_split으로 8:2 분리
   - LogisticRegression 학습
   - classification_report 출력
   - joblib으로 {"clf": clf, "le": le} 저장
   - clf, le 반환
   """
   label_encoder = LabelEncoder()
   y_encoded = label_encoder.fit_transform(y)

   print(f"[{datetime.now():%H:%M:%S}] [classifier] 클래스 매핑: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

   # train / test 분리 (8:2)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y_encoded,
       test_size=test_size,
       random_state=random_state,
       stratify=y_encoded,
   )

   # 학습
   clf = LogisticRegression(
       max_iter=1000,
       random_state=random_state,
       class_weight='balanced' # imbalanced class로 인해 소수 클래스에 가중치 부여 추가
   )
   clf.fit(X_train, y_train)

   # 평가
   y_pred = clf.predict(X_test)
   print("\n[{datetime.now():%H:%M:%S}] [classifier] === Classification Report ===")
   print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

   # 저장
   payload = {"clf": clf, "le": label_encoder}
   joblib.dump(payload, model_save_path)
   print(f"[{datetime.now():%H:%M:%S}] [classifier] 모델 저장 완료: {model_save_path}")

   return clf, label_encoder


def load_classifier(model_path):
    """
    - joblib으로 모델 불러와서 clf, le 반환
    """
    payload = joblib.load(model_path)
    classifier = payload["clf"]
    label_encoder = payload["le"]
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 모델 로드 완료: {model_path}")
    print(f"[{datetime.now():%H:%M:%S}] [classifier] 클래스: {list(label_encoder.classes_)}")
    return clf, label_encoder

def predict_segment(frames, clf, le, confidence_threshold, clip_model):
    """
    - 프레임 리스트를 받아서 임베딩 추출
    - clf.predict_proba로 확률 계산
    - 최대 확률이 confidence_threshold 미만이면 UNCERTAIN 반환
    - (BehaviourClass, confidence) 튜플 반환
    """
    if not frames:
        print(f"[{datetime.now():%H:%M:%S}] [classifier] 빈 프레임 → UNCERTAIN 반환")
        return BehaviourClass.UNCERTAIN, 0.0

    try:
        embedding = frames_to_embedding(frames, model_name=clip_model)
        embedding_2d = embedding.reshape(1, -1)  # (1, 512)

        proba = clf.predict_proba(embedding_2d)[0]  # (n_classes,)
        max_confidence = float(proba.max())
        predicted_idx = int(proba.argmax())

        if max_confidence < confidence_threshold:
            return BehaviourClass.UNCERTAIN, max_confidence

        label_str = le.inverse_transform([predicted_idx])[0] # number to string
        behaviour = BehaviourClass(label_str) # string to enum

        return behaviour, max_confidence

    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] [classifier] predict_segment 오류: {e}")
        return BehaviourClass.UNCERTAIN, 0.0

X, y = build_feature_matrix("../data/duck.mp4", "../data/labels.csv", "ViT-B/32")
clf, label_encoder= train_classifier(X, y, "../models/classifier.joblib")
# df = pd.read_csv( "../data/labels.csv", index_col=False)
# print(df.columns)
# print(df.head())