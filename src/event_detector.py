import datetime
from collections import deque

from src.behaviour_states import BehaviourClass


class EventDetector:
    def __init__(self, smoothing_window, max_uncertain_streak):
        self.smoothing_window = smoothing_window
        self.max_uncertain_streak = max_uncertain_streak
        self._history = deque(maxlen=smoothing_window * 2)
        self._stable_behaviour = None
        self._uncertain_streak = 0
        self.events = []

    def update(self, behaviour, confidence, start_sec):
        self._history.append(
            {
                "behaviour": behaviour,
                "confidence": confidence,
                "start_sec": start_sec,
            }
        )

        if behaviour == BehaviourClass.UNCERTAIN:
            self._uncertain_streak += 1
        else:
            self._uncertain_streak = 0

        recent = list(self._history)[-self.smoothing_window :]
        if len(recent) < self.smoothing_window:
            return None

        behaviours = [
            item["behaviour"]
            for item in recent
            if item["behaviour"] != BehaviourClass.UNCERTAIN
        ]

        if not behaviours:
            return None

        new_behaviour = behaviours[0]
        if all(item == new_behaviour for item in behaviours):
            if new_behaviour != self._stable_behaviour:
                previous_behaviour = self._stable_behaviour
                self._stable_behaviour = new_behaviour
                event = {
                    "type": "transition",
                    "from": previous_behaviour.value if previous_behaviour else None,
                    "to": new_behaviour.value,
                    "at_sec": start_sec,
                    "at_time": self._format_time(start_sec),
                    "confidence": confidence,
                }
                self.events.append(event)
                return event

        return None

    def should_play_uncertain_sound(self):
        return self._uncertain_streak == self.max_uncertain_streak

    def get_current_stable(self):
        return self._stable_behaviour

    def get_top_events(self, n):
        return [event for event in self.events if event.get("type") == "transition"][:n]

    @staticmethod
    def _format_time(start_sec):
        total_seconds = int(datetime.timedelta(seconds=start_sec).total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02}"
