import os
import random
from src.behaviour_states import BehaviourClass, BEHAVIOUR_MESSAGES

BEHAVIOUR_FACE_MAP = {
    BehaviourClass.RESTING: "faces/idle",
    BehaviourClass.FEEDING: "faces/thinking",
    BehaviourClass.EXPLORING: "faces/speaking",
    BehaviourClass.UNCERTAIN: "faces/warmup",
}

def get_face_images(behaviour):
    folder_path = BEHAVIOUR_FACE_MAP.get(behaviour, "faces/idle")
    image_paths = _get_png_paths(folder_path)

    if not image_paths:
        image_paths = _get_png_paths("faces/idle")

    return image_paths


def get_reaction_message(behaviour):
    messages = BEHAVIOUR_MESSAGES[behaviour]
    return random.choice(messages)


def get_uncertain_sound(sound_dir):
    if not os.path.isdir(sound_dir):
        return None

    wav_paths = sorted(
        os.path.join(sound_dir, filename)
        for filename in os.listdir(sound_dir)
        if filename.lower().endswith(".wav")
    )
    if not wav_paths:
        return None

    return random.choice(wav_paths)


def _get_png_paths(folder_path):
    if not os.path.isdir(folder_path):
        return []

    return sorted(
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(".png")
    )
