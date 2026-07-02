# 🦆 be-more-agent-duck

> Raspberry Pi-based lightweight embodied agent for duck behaviour observation.

[![Original Project](https://img.youtube.com/vi/l5ggH-YhuAw/maxresdefault.jpg)](https://youtu.be/l5ggH-YhuAw)

`be-more-agent-duck` is a **fork-based** refactoring project that changes the original offline voice assistant into a duck behaviour observation agent.
The system takes a duck video as input, splits it into short segments, classifies behaviour, reacts to behaviour changes, and saves a session report.

---

## Overview

This project focuses on a lightweight behaviour observation pipeline rather than a general-purpose voice assistant.

The MVP goal is:

```text
Duck video
  → 5-second segment split
  → 10-frame sampling per segment
  → CLIP image embedding average
  → Logistic Regression behaviour classification
  → behaviour transition detection
  → character reaction
  → session report
  → CSV / JSON logging
```

The current target behaviour classes are:

| Class       | Meaning                                           | Visual cue                                    |
| ----------- | ------------------------------------------------- | --------------------------------------------- |
| `resting`   | The duck is staying still or resting              | Little movement, compact posture              |
| `feeding`   | The duck is eating or dipping its head into water | Repeated head-down movement                   |
| `exploring` | The duck is walking, swimming, or moving around   | Movement, direction change                    |
| `uncertain` | Confidence is too low to classify reliably        | Automatically assigned when confidence is low |

---

## Motivation

Many behaviour recognition systems stop at classification.
This project tries to make the result more interpretable by adding an embodied interface:

* observe behaviour in short video segments
* react when the behaviour changes
* summarize the session as a timeline and report
* save structured records for later analysis

The system is intentionally lightweight so it can later run on a Raspberry Pi 5 as an offline demonstration.

---

## Key Features

### 1. Segment-based behaviour observation

The video is divided into fixed-length segments.
Each segment is sampled into a small number of frames and classified into one of the target behaviour states.

Default parameters:

| Parameter              | Default | Meaning                                                            |
| ---------------------- | ------: | ------------------------------------------------------------------ |
| `segment_sec`          |     `5` | Segment length in seconds                                          |
| `frames_per_seg`       |    `10` | Number of sampled frames per segment                               |
| `confidence_threshold` |   `0.6` | If confidence is lower than this, classify as `uncertain`          |
| `smoothing_window`     |     `3` | Number of stable segments required before detecting transition     |
| `max_uncertain_streak` |     `3` | Number of consecutive uncertain segments before uncertain feedback |

### 2. CLIP embedding + Logistic Regression baseline

The first ML baseline uses:

* CLIP image encoder as a feature extractor
* average embedding across sampled frames
* Logistic Regression as the classifier

This is selected for fast prototyping, interpretability, and Raspberry Pi portability.

### 3. Behaviour transition reaction

The system does not react to every single prediction.
Instead, it waits until a behaviour is stable for several segments and then reacts when the stable behaviour changes.

Example:

```text
resting → feeding
feeding → exploring
exploring → resting
```

### 4. Session report and logging

At the end of a session, the system generates:

* behaviour timeline
* representative transition events
* short summary text
* `session.csv`
* `session.json`

The output is saved under the `sessions/` directory.
`EventDetector.get_top_events(5)` is responsible for selecting transition events.
`generate_report()` receives those selected events and writes them into the report.

---

## Project Structure

Current target structure:

```text
be-more-agent-duck/
|
|-- agent.py                   # Entry point, currently being refactored
|-- config.json                # Runtime parameters
|-- requirements.txt           # Python dependencies
|-- setup.sh                   # Setup script
|-- README.md                  # Project documentation
|
|-- src/
|   |-- behaviour_states.py    # Behaviour class enum and messages
|   |-- video_processor.py     # Video loading and frame sampling
|   |-- classifier.py          # CLIP embedding and Logistic Regression
|   |-- event_detector.py      # Behaviour transition detection
|   |-- reaction.py            # Character face/message/sound reaction logic
|   |-- report_generator.py    # Session report generation
|   |-- session_logger.py      # CSV / JSON session saving
|   `-- __init__.py
|
|-- gui.py                     # DuckGUI placeholder
|
|-- faces/
|   |-- capturing/
|   |-- error/
|   |-- idle/
|   |-- listening/
|   |-- speaking/
|   |-- thinking/
|   `-- warmup/
|
|-- sounds/
|   |-- ack_sounds/
|   |-- greeting_sounds/
|   `-- thinking_sounds/       # Used for uncertain feedback
|
|-- data/
|   |-- duck1.mp4
|   |-- duck2.mp4
|   |-- labels_duck1.csv
|   `-- labels_duck2.csv
|
|-- sessions/
|   `-- YYYY-MM-DD_HH-MM-SS/
|       |-- session.csv
|       `-- session.json
|
|-- models/
|   `-- classifier.joblib
|
`-- notebooks/
    `-- auto_label.py
```

The `faces/` folders are BMO expression states, not one folder per duck behaviour.
`src/reaction.py` maps duck behaviours to these expression folders.

---

## Dataset and Label Format

Training data can be combined from multiple videos and matching label CSV files.
The current local dataset uses:

```text
data/duck1.mp4          + data/labels_duck1.csv
data/duck2.mp4          + data/labels_duck2.csv
```

Example:

```csv
segment_id,start_sec,end_sec,label,notes
0,0,5,resting,staying still near water
1,5,10,exploring,walking to the left
2,10,15,feeding,dipping head into water
3,15,20,resting,
4,20,25,uncertain,unclear movement
```

Initial labeling target:

* at least 20 segments for `resting`
* at least 20 segments for `feeding`
* at least 20 segments for `exploring`
* `uncertain` can be used for unclear sections but is excluded from classifier training

---

## Configuration

Example `config.json`:

```json
{
  "video_path": "data/duck1.mp4",
  "segment_sec": 5,
  "frames_per_seg": 10,
  "confidence_threshold": 0.6,
  "smoothing_window": 3,
  "max_uncertain_streak": 3,
  "output_dir": "sessions",
  "model_path": "models/classifier.joblib",
  "clip_model": "ViT-B/32",
  "uncertain_sound_dir": "sounds/thinking_sounds"
}
```

When `config.json` exists, runtime paths and parameters are controlled from that file.
Code-level defaults are only a fallback for a missing config file.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MelonChicken/be-more-agent-duck.git
cd be-more-agent-duck
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare data

Place duck video files under `data/`, for example:

```text
data/duck1.mp4
data/duck2.mp4
```

Then create or edit matching label CSV files:

```text
data/labels_duck1.csv
data/labels_duck2.csv
```

---

## Training Baseline Model

The current training helper supports multiple video/label pairs:

```python
from src.classifier import build_feature_matrix, train_classifier

X, y = build_feature_matrix(
    ["data/duck1.mp4", "data/duck2.mp4"],
    ["data/labels_duck1.csv", "data/labels_duck2.csv"],
    "ViT-B/32",
    frames_per_seg=10,
)
clf, label_encoder = train_classifier(X, y, "models/classifier.joblib")
```

The training path and inference path both sample 10 frames per 5-second segment.
This keeps CLIP embedding extraction fast and avoids train/inference sampling skew.

If `build_feature_matrix(..., cache_path=...)` is used, delete and regenerate the `.npz`
cache whenever `frames_per_seg` or sampling logic changes. A cache generated from a
different frame count is not valid for the current 10-frame baseline.

The `notebooks/auto_label.py` script can be used as a helper during dataset preparation.

---

## Running the Agent

After the classifier is trained, run the headless pipeline:

```bash
python agent.py --headless
```

This processes the configured video, prints the summary, and saves CSV/JSON outputs.
The GUI path will use the same `DuckAgent.run(on_segment=...)` callback contract once
`DuckGUI` is implemented.
At session end, results are saved to the `sessions/` directory.

---

## MVP Success Criteria

The MVP is considered complete when:

* a 2–3 minute duck video can be processed end-to-end
* segment-level behaviour classification works
* behaviour transition events are detected
* the GUI displays behaviour state and reaction text
* session report is generated
* `session.csv` and `session.json` are saved
* validation accuracy is at least 70%
* uncertain ratio is 30% or lower

---

## Roadmap

### MVP

* [x] Clean up original voice-assistant files
* [x] Replace `config.json`
* [x] Replace `requirements.txt`
* [x] Add video processing module
* [x] Add CLIP + Logistic Regression classifier
* [x] Use 10-frame sampling for training embeddings
* [x] Add event detector
* [x] Add reaction logic
* [x] Add session logger
* [x] Add report generator
* [x] Integrate modules in `agent.py`
* [ ] Refactor GUI into `DuckGUI`
* [ ] Run end-to-end test with multi-video training data

### After MVP

* [ ] Run offline on Raspberry Pi 5
* [ ] Add daily behaviour pattern visualization
* [ ] Add more behaviour classes such as `preening` and `swimming`
* [ ] Connect real-time camera input
* [ ] Improve temporal smoothing

---

## Current Status

This repository is currently under refactoring.

The original project was an offline voice assistant using wake word detection, speech-to-text, a local LLM, text-to-speech, and reactive face animations.
This fork is being changed into a video-based duck behaviour observation agent.

Some original files may still exist temporarily during the refactoring process.

---

## Attribution

This project is a fork of `brenpoly/be-more-agent`.

Original project:

* Repository: `brenpoly/be-more-agent`
* Copyright: Copyright (c) 2026 brenpoly
* License: MIT License

This fork keeps the original MIT license notice while modifying the project direction, architecture, and runtime pipeline for duck behaviour observation.

---

## License

The software code is licensed under the MIT License.

See the `LICENSE` file for the full license text.

If original non-code assets are reused, such as character images, sound effects, or 3D model files, check the asset-specific license terms before redistribution.
