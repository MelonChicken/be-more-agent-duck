import clip
import torch
import numpy as np
import pandas as pd
import joblib
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
    return mean_embedding.cpu().numpy()


def build_feature_matrix(video_path, labels_csv,):
    """
    - labels.csv를 pandas로 읽어라
    - label == "uncertain" 인 행은 제외해라
    - 각 구간의 프레임을 추출하고 임베딩을 뽑아서
    - X (numpy array), y (label 문자열 array) 반환
    """

def train_classifier(X, y, model_save_path):
   """
   - LabelEncoder로 y를 숫자로 변환
   - train_test_split으로 8:2 분리
   - LogisticRegression 학습
   - classification_report 출력
   - joblib으로 {"clf": clf, "le": le} 저장
   - clf, le 반환
   """

def load_classifier(model_path):
   """
   - joblib으로 모델 불러와서 clf, le 반환
   """

def predict_segment(frames, clf, le, confidence_threshold, clip_model):
   """
   - 프레임 리스트를 받아서 임베딩 추출
   - clf.predict_proba로 확률 계산
   - 최대 확률이 confidence_threshold 미만이면 UNCERTAIN 반환
   - (BehaviourClass, confidence) 튜플 반환
   """

for seg_id, start_sec, frames in iter_segments("../data/duck.mp4", 5, 10):
    frames_to_embedding(frames, "ViT-B/32")