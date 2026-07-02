import os
import tkinter as tk

from PIL import Image, ImageTk

from src.behaviour_states import BehaviourClass
from src.reaction import get_face_images, get_reaction_message


class DuckGUI:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.root = tk.Tk()
        self.root.title("Be More Agent Duck")
        self.root.geometry("800x480")
        self.root.configure(bg="#111111")

        self._queue = []
        self._idx = 0
        self._report = None
        self._current_img = None

        self.image_label = tk.Label(self.root, bg="#111111")
        self.image_label.pack(pady=(28, 16))

        self.caption_label = tk.Label(
            self.root,
            text="",
            bg="#111111",
            fg="#f4f4f4",
            font=("Arial", 18),
            wraplength=720,
            justify="center",
        )
        self.caption_label.pack(pady=(0, 18))

        self.status_label = tk.Label(
            self.root,
            text="Ready",
            bg="#111111",
            fg="#b8b8b8",
            font=("Arial", 13),
        )
        self.status_label.pack()

    def start(self):
        self.status_label.config(text="Analyzing video...")
        self.root.after(100, self._analyze_then_play)
        self.root.mainloop()

    def _analyze_then_play(self):
        self._report = self.agent.run(on_segment=self._collect)
        self._idx = 0
        self._play_next()

    def run(self):
        return self.start()

    def _collect(self, seg_info):
        self._queue.append(seg_info)

    def _play_next(self):
        if self._idx >= len(self._queue):
            self._show_report()
            return

        seg_info = self._queue[self._idx]
        self._idx += 1

        self._update_image(seg_info)
        self._update_text(seg_info)

        playback_speed = self.config.get("playback_speed", 8)
        interval_ms = max(1, int(self.config["segment_sec"] * 1000 / playback_speed))
        self.root.after(interval_ms, self._play_next)

    def _update_image(self, seg_info):
        image_paths = seg_info.get("face_images") or self._get_face_images(seg_info)
        if not image_paths:
            self.image_label.config(image="", text="No image", fg="#f4f4f4")
            self._current_img = None
            return

        image_path = image_paths[(self._idx - 1) % len(image_paths)]
        if not os.path.exists(image_path):
            self.image_label.config(image="", text="Image missing", fg="#f4f4f4")
            self._current_img = None
            return

        image = Image.open(image_path)
        image.thumbnail((280, 280))
        self._current_img = ImageTk.PhotoImage(image)
        self.image_label.config(image=self._current_img, text="")

    def _update_text(self, seg_info):
        behaviour = seg_info["behaviour"]
        confidence = seg_info["conf"]
        start = seg_info["start"]
        message = seg_info.get("message") or self._get_reaction_message(seg_info)

        event = seg_info.get("event")
        if event:
            event_text = f" | transition: {event['from']} -> {event['to']}"
            status_color = "#ffd166"
        else:
            event_text = ""
            status_color = "#b8b8b8"

        self.caption_label.config(text=message)
        self.status_label.config(
            text=f"{start:>5}s | {behaviour} | conf {confidence:.2f}{event_text}",
            fg=status_color,
        )

    def _show_report(self):
        summary = self._report["summary"] if self._report else "No report generated."
        self.image_label.config(image="", text="Session complete", fg="#f4f4f4")
        self._current_img = None
        self.caption_label.config(text=summary)
        self.status_label.config(text="Report saved", fg="#b8f2bb")

    def _get_face_images(self, seg_info):
        try:
            behaviour = BehaviourClass(seg_info["behaviour"])
        except ValueError:
            return get_face_images(BehaviourClass.UNCERTAIN)
        return get_face_images(behaviour)

    def _get_reaction_message(self, seg_info):
        try:
            behaviour = BehaviourClass(seg_info["behaviour"])
        except ValueError:
            behaviour = BehaviourClass.UNCERTAIN
        return get_reaction_message(behaviour)
