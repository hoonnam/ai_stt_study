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





# Whisper: OpenAI's Automatic Speech Recognition (ASR) Model

Whisper is an automatic speech recognition (ASR) model developed by OpenAI and released in September 2022. It is designed to convert spoken language into text with high accuracy, leveraging a Transformer-based encoder-decoder architecture.

## Features
- **Multilingual Support**: Recognizes over 99 languages and includes automatic language detection.
- **Versatile Capabilities**:
  - Speech-to-text (STT)
  - Translation (between English and other languages)
  - Sentence-level timestamps
  - Robust performance in noisy environments
- **Zero-shot Learning**: Generalizes well to various tasks without additional fine-tuning.
- **Extensive Training Data**: Trained on approximately 680,000 hours of audio data for high adaptability.

## Model Architecture
Whisper utilizes a Transformer-based approach and employs special tokens for different tasks:
- `⟨|transcribe|⟩`: Specifies text transcription.
- `⟨|translate|⟩`: Specifies translation tasks.
- `⟨|notimestamps|⟩`: Disables timestamp generation.

## Use Cases
- Transcription of podcasts, lectures, and meetings.
- Subtitle generation for multilingual media.
- Voice-command-based interfaces.
- Enhanced audio data indexing and searchability.

## Model Sizes and Performance
Whisper provides multiple model sizes based on computational resources:

| Size     | Parameters | Multilingual Model | Required VRAM | Relative Speed |
|----------|-----------|--------------------|--------------|---------------|
| Tiny     | 39M       | `tiny`             | ~1GB         | ~32x          |
| Base     | 74M       | `base`             | ~1GB         | ~16x          |
| Small    | 244M      | `small`            | ~2GB         | ~6x           |
| Medium   | 769M      | `medium`           | ~5GB         | ~2x           |
| Large    | 1550M     | `large`            | ~10GB        | 1x            |

## Installation & Usage
Whisper can be easily installed and used in a Python environment:

```bash
pip install openai-whisper
```

### Example Code:
```python
import whisper

# Load the model
model = whisper.load_model("base")

# Transcribe an audio file
result = model.transcribe("audio_file_path.mp3")

# Print the transcription
print(result["text"])
```

## Conclusion
Whisper is a powerful tool for speech recognition, offering high accuracy and flexibility. Its multilingual support and zero-shot learning capability make it an excellent choice for various applications across industries.


