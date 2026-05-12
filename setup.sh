#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🦆 be-more-agent-duck Setup Script${NC}"

# 1. System Dependencies
echo -e "${YELLOW}[1/4] Installing system tools...${NC}"
sudo apt update
sudo apt install -y python3-tk python3-dev cmake build-essential git \
                    libgl1-mesa-glx libglib2.0-0

# 2. Create Folders
echo -e "${YELLOW}[2/4] Creating folders...${NC}"
mkdir -p data/clips
mkdir -p sessions
mkdir -p models
mkdir -p notebooks
mkdir -p sounds/uncertain_sounds

# 3. Python Libraries
echo -e "${YELLOW}[3/4] Installing Python libraries...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
echo -e "${YELLOW}[4/4] Verifying install...${NC}"
python3 -c "import clip;       print('[OK] CLIP')"       || echo -e "${RED}[FAIL] CLIP${NC}"
python3 -c "import cv2;        print('[OK] OpenCV')"     || echo -e "${RED}[FAIL] OpenCV${NC}"
python3 -c "import torch;      print('[OK] torch')"      || echo -e "${RED}[FAIL] torch${NC}"
python3 -c "import sklearn;    print('[OK] scikit-learn')"  || echo -e "${RED}[FAIL] scikit-learn${NC}"

echo -e "${GREEN}✨ Setup complete! Run: python agent.py${NC}"