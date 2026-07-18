import argparse
import json
import os
from collections import deque

from src.behaviour_states import BehaviourClass, DEFAULT_CONFIDENCE_THRESHOLD
from src.classifier import get_clip_model, predict_segment_proba, load_or_train_classifier
from src.event_detector import EventDetector
from src.reaction import get_face_images, get_reaction_message, get_uncertain_sound
from src.report_generator import generate_report
from src.session_logger import SessionLogger
from src.video_processor import iter_segments, load_video


DEFAULT_CONFIG = {
    "video_path": "data/test/duck1.mp4",
    "segment_sec": 5,
    "playback_speed": 8,
    "frames_per_seg": 10,
    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "confidence_margin_threshold": 0.15,
    "prediction_smoothing_window": 3,
    "smoothing_window": 3,
    "max_uncertain_streak": 3,
    "output_dir": "sessions",
    "model_path": "models/classifier.joblib",
    "clip_model": "ViT-B/32",
    "uncertain_sound_dir": "sounds/thinking_sounds",
    "train_labels_dir": "data/wetlandbirds",
    "train_boxes_csv": "data/wetlandbirds/duck_segment_boxes.csv",
    "train_videos_dir": "data/wetlandbirds/videos",
    "test_labels": ["data/labels_duck1.csv", "data/labels_duck2.csv"],
}


def load_config(path="config.json"):
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DuckAgent:
    def __init__(self, config):
        self.config = config
        self.classifier, self.label_encoder = load_or_train_classifier(model_path = config["model_path"], config = config)
        self.detector = EventDetector(
            config["smoothing_window"],
            config["max_uncertain_streak"],
        )
        self.session_logger = SessionLogger(config["output_dir"])
        self.clip_model = config["clip_model"]
        self.confidence_threshold = config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        self.confidence_margin_threshold = config.get("confidence_margin_threshold", 0.15)
        self.prediction_smoothing_window = max(1, int(config.get("prediction_smoothing_window", 1)))
        self._proba_history = deque(maxlen=self.prediction_smoothing_window)
        self._none_streak = 0
        get_clip_model(self.clip_model)

    def analyze_segment(self, frames):
        proba = predict_segment_proba(
            frames,
            self.classifier,
            self.label_encoder,
            self.clip_model,
        )
        if proba is None:
            self._none_streak += 1
            if self._none_streak >= self.prediction_smoothing_window:
                self._proba_history.clear()
            return BehaviourClass.UNCERTAIN, 0.0

        self._none_streak = 0
        self._proba_history.append(proba)
        smoothed_proba = sum(self._proba_history) / len(self._proba_history)
        sorted_proba = sorted(smoothed_proba, reverse=True)
        confidence = float(sorted_proba[0])
        margin = float(sorted_proba[0] - sorted_proba[1]) if len(sorted_proba) > 1 else confidence

        if confidence < self.confidence_threshold and margin < self.confidence_margin_threshold:
            return BehaviourClass.UNCERTAIN, confidence

        predicted_idx = int(smoothed_proba.argmax())
        label_str = self.label_encoder.inverse_transform([predicted_idx])[0]
        return BehaviourClass(label_str), confidence

    def run(self, on_segment=None):
        timeline = []
        total_sec = self._get_total_sec()

        for seg_id, start_sec, frames in iter_segments(
            self.config["video_path"],
            self.config["segment_sec"],
            self.config["frames_per_seg"],
        ):
            behaviour, confidence = self.analyze_segment(frames)
            event = self.detector.update(behaviour, confidence, start_sec)

            timeline_entry = {
                "seg": seg_id,
                "start": start_sec,
                "behaviour": behaviour.value,
                "conf": confidence,
            }
            timeline.append(timeline_entry)
            self.session_logger.record_segment(timeline_entry)

            segment_info = {
                **timeline_entry,
                "message": get_reaction_message(behaviour),
                "face_images": get_face_images(behaviour),
                "event": event,
            }
            if self.detector.should_play_uncertain_sound():
                segment_info["sound"] = get_uncertain_sound(
                    self.config["uncertain_sound_dir"]
                )

            if on_segment is not None:
                on_segment(segment_info)

        report = generate_report(
            timeline,
            self.detector.get_top_events(5),
            total_sec,
        )
        self.session_logger.save(report, timeline)
        return report

    def _get_total_sec(self):
        cap, fps, total_frames, total_sec = load_video(self.config["video_path"])
        cap.release()
        return total_sec


def main():
    parser = argparse.ArgumentParser(description="Duck behaviour observation agent")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    agent = DuckAgent(config)

    if args.headless:
        report = agent.run()
        print(report["summary"])
        return report

    return _run_gui(agent, config)


def _run_gui(agent, config):
    try:
        from gui import DuckGUI
    except ImportError as exc:
        raise RuntimeError(
            "DuckGUI is not available yet. Run with --headless or implement gui.DuckGUI."
        ) from exc

    gui = DuckGUI(agent, config)
    return gui.start()


if __name__ == "__main__":
    main()
