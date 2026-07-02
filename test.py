from src.behaviour_states import BehaviourClass
from src.classifier import build_feature_matrix, train_classifier
from src.event_detector import EventDetector
from src.reaction import get_face_images, get_reaction_message, get_uncertain_sound
from src.report_generator import generate_report

# X, y = build_feature_matrix(["data/duck1.mp4", "data/duck2.mp4"], ["data/labels_duck1.csv", "data/labels_duck2.csv"], "ViT-B/32")
# clf, label_encoder= train_classifier(X, y, "models/classifier.joblib")

detector = EventDetector(smoothing_window=3, max_uncertain_streak=3)

# 같은 행동 3번 연속
detector.update(BehaviourClass.FEEDING, 0.85, 0.0)
detector.update(BehaviourClass.FEEDING, 0.80, 5.0)
event = detector.update(BehaviourClass.FEEDING, 0.82, 10.0)
print(event)  # {"type": "transition", "from": None, "to": "feeding", ...}

# 다른 행동으로 전환
detector.update(BehaviourClass.RESTING, 0.75, 15.0)
detector.update(BehaviourClass.RESTING, 0.78, 20.0)
event = detector.update(BehaviourClass.RESTING, 0.80, 25.0)
print(event)  # {"type": "transition", "from": "feeding", "to": "resting", ...}

print(get_face_images(BehaviourClass.FEEDING))
# ["faces/thinking/thinking 01.png", "faces/thinking/thinking 02.png", ...]

print(get_reaction_message(BehaviourClass.RESTING))
# "쉬고 있어요 🌙"  (랜덤)

print(get_uncertain_sound())
# "sounds/uncertain_sounds/computing.wav"  (랜덤)

timeline = [
    {"seg": 0, "start": 0, "behaviour": "feeding", "conf": 0.85},
    {"seg": 1, "start": 5, "behaviour": "feeding", "conf": 0.80},
    {"seg": 2, "start": 10, "behaviour": "resting", "conf": 0.72},
]
report = generate_report(timeline, events=[], total_sec=15)
print(report["summary"])
# "이번 세션에서 오리는 주로 'feeding' 상태였으며, 전체 구간의 약 66%를 차지했습니다. ..."