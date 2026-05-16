import os
import json
import csv
import datetime


def get_session_dir(base_dir):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(base_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def save_session_csv(timeline, session_dir):
    if not timeline:
        return
    filepath = os.path.join(session_dir, "session.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
        writer.writeheader()
        writer.writerows(timeline)


def save_session_json(report, session_dir):
    filepath = os.path.join(session_dir, "session.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_session(timeline, report, base_dir="sessions"):
    session_dir = get_session_dir(base_dir)
    save_session_csv(timeline, session_dir)
    save_session_json(report, session_dir)
    return session_dir