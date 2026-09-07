# EyeQ

EyeQ turns your webcam into a hand-gesture-controlled drawing and vision assistant. Draw or box an object in the air using just your hand, then ask an AI what it's looking at 

## What it does

- **Draw mode**: point one finger at the camera to draw on screen, like a virtual whiteboard.
- **Box mode**: point one finger to drag out a box around something in the camera view.
- **Ask the AI**: pinch your thumb and index finger together to send your drawing (or boxed area) to a vision model, ask it a question out loud, and hear the answer spoken back.
- **Other gestures**:
  - Open palm — clear the canvas
  - Fist — cancel/interrupt the AI mid-response
  - Thumbs up — switch between Draw and Box mode
  - Three fingers up — 3-2-1 countdown, then save a snapshot
  - Two hands — move apart/together to zoom in and out
- A short on-screen tutorial walks you through each gesture the first time you run it.

## How it works

- **Hand tracking**: [MediaPipe](https://github.com/google/mediapipe) detects your hand landmarks from the webcam feed in real time.
- **Vision**: your drawing/box crop is sent to either Anthropic's Claude or Featherless AI (whichever key you provide) to answer questions about it.
- **Voice**: your spoken question is transcribed with `SpeechRecognition` (Google's speech API), and the AI's answer is read back out loud with `pyttsx3` (offline text-to-speech).

## Setup

### Requirements

- Python 3.10+
- A webcam and a microphone
- An API key for **either**:
  - [Anthropic](https://console.anthropic.com) (Claude), **or**
  - [Featherless AI](https://featherless.ai)

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

On first run, if no API key is found, you'll be prompted in the terminal to enter one. It's saved to a `.env` file next to the script so you won't be asked again.

You can also create the `.env` file yourself ahead of time:

```
ANTHROPIC_API_KEY=your-key-here
FEATHERLESS_API_KEY=your-key-here
```

Only one key is required — you'll get a setup screen to pick which model to use if both are present.
