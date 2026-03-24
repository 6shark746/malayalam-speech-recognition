# Real-Time Malayalam Speech Recognition

This project implements a real-time Malayalam speech recognition system using a fine-tuned Whisper Medium model.

## Features
- Real-time speech-to-text
- Malayalam language support
- Confidence score calculation
- Performance metrics (WER, BER, Accuracy)

## Model
Fine-tuned Whisper Medium model using 240 samples from Indic TTS Malayalam Speech Corpus.

## Tech Stack
- Python
- WhisperX
- NumPy
- SoundDevice
- JiWER

## How to Run

```bash
pip install -r requirements.txt
python main.py

---

# 📦 4. Create `requirements.txt`

Run this in your terminal:

```bash
pip freeze > requirements.txt
## Results
- Confidence: ~88%
- WER: ~0.30