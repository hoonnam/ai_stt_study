# Real-Time Korean Speech-to-Text with Whisper

## Overview
This project utilizes OpenAI's Whisper model to transcribe Korean speech into text in real-time. Several optimization techniques have been applied to improve speed and performance.

## Features
- Real-time Korean speech recognition
- Silence detection to reduce unnecessary processing
- Asynchronous processing using queues and threads
- Optimized Whisper parameters for speed
- Supports GPU acceleration (CUDA)

## Installation

### Requirements
- Python 3.8+
- Required packages:
  ```bash
  pip install sounddevice numpy whisper torch scipy
  ```
- Ensure that your system has a working microphone.

## How It Works
The program consists of five main functions and a `main` function:

### 1. `preprocess_audio(audio_data)`
- Normalizes the audio by adjusting the mean to zero and scaling to the maximum absolute value.
- **Optimization:** Performs minimal preprocessing to reduce processing time.

### 2. `is_silent(audio_data)`
- Detects silence by calculating the RMS (Root Mean Square) value of the audio.
- **Optimization:** Uses RMS for fast silence detection.

### 3. `audio_callback(indata, frames, time, status)`
- Captures microphone input and stores it in a queue.
- **Optimization:** Uses a queue to process data asynchronously.

### 4. `process_audio_chunk(audio_chunk)`
- Converts audio chunks into text using the Whisper model.
- **Optimization:**
  - Uses `temperature=0.0`, `beam_size=1`, and `best_of=1` for speed.
  - Enables `fp16=True` to reduce memory usage and processing time.

### 5. `transcribe_audio()`
- Fetches audio data from the queue and transcribes it in real-time.
- **Optimization:**
  - Skips silent sections.
  - Avoids duplicate outputs to reduce unnecessary processing.

### 6. `main()`
- Manages audio input, transcription, and output.
- Runs `transcribe_audio` in a separate thread for asynchronous processing.
- **Optimization:**
  - Uses threading to minimize latency.

## Optimization Techniques
- Lightweight audio preprocessing
- RMS-based silence detection
- Asynchronous processing with a queue
- Whisper parameter tuning for speed
- FP16 (float16) for memory efficiency
- Skipping silent and redundant text outputs
- Multi-threading to minimize delays

## Running the Program
To start the real-time transcription:
```bash
python stt.py
```
Press `Ctrl+C` to exit the program.



