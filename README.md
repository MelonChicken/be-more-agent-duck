# 🦆 be-more-agent-duck

> Raspberry Pi-based lightweight embodied agent for duck behaviour observation.

`be-more-agent-duck` is a fork-based refactoring project that changes the original offline voice assistant into a duck behaviour observation agent.
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

---

## Project Structure

Planned target structure:

```text
be-more-agent-duck/
│
├── agent.py                   # Entry point
├── config.json                # Runtime parameters
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
├── README.md                  # Project documentation
│
├── behaviour_states.py        # Behaviour class enum and messages
├── video_processor.py         # Video loading and frame sampling
├── classifier.py              # CLIP embedding and Logistic Regression
├── event_detector.py          # Behaviour transition detection
├── reaction.py                # Character face/message/sound reaction logic
├── report_generator.py        # Session report generation
├── session_logger.py          # CSV / JSON session saving
├── gui.py                     # DuckGUI interface
│
├── faces/
│   ├── resting/
│   ├── feeding/
│   ├── exploring/
│   ├── uncertain/
│   ├── idle/
│   └── error/
│
├── sounds/
│   └── uncertain_sounds/
│
├── data/
│   ├── labels.csv
│   └── clips/
│
├── sessions/
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── session.csv
│       └── session.json
│
├── models/
│   └── duck_classifier.pkl
│
└── notebooks/
    └── baseline.ipynb
```

---

## Dataset and Label Format

Training data is managed with `data/labels.csv`.

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
  "video_path": "data/duck.mp4",
  "segment_sec": 5,
  "frames_per_seg": 10,
  "confidence_threshold": 0.6,
  "smoothing_window": 3,
  "max_uncertain_streak": 3,
  "output_dir": "sessions",
  "model_path": "models/duck_classifier.pkl",
  "clip_model": "ViT-B/32",
  "uncertain_sound_dir": "sounds/uncertain_sounds"
}
```

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

Place a duck video file at:

```text
data/duck.mp4
```

Then create or edit:

```text
data/labels.csv
```

---

## Training Baseline Model

The initial training workflow is expected to be run in:

```text
notebooks/baseline.ipynb
```

The notebook should:

1. load `config.json`
2. read `data/labels.csv`
3. extract CLIP embeddings from labeled segments
4. train Logistic Regression
5. save the model to `models/duck_classifier.pkl`

---

## Running the Agent

After the classifier is trained:

```bash
python agent.py
```

The GUI starts the video-based behaviour observation pipeline.
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

* [ ] Clean up original voice-assistant files
* [ ] Replace `config.json`
* [ ] Replace `requirements.txt`
* [ ] Add video processing module
* [ ] Add CLIP + Logistic Regression classifier
* [ ] Add event detector
* [ ] Add reaction logic
* [ ] Add session logger
* [ ] Add report generator
* [ ] Refactor GUI into `DuckGUI`
* [ ] Run end-to-end test with one duck video

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
