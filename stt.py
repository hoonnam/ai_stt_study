import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import torch

# GPU 가속 활성화
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Whisper 모델 로드 (더 작은 모델 사용)
model = whisper.load_model("base").to(device)

# 오디오 설정
SAMPLE_RATE = 16000
BUFFER_SIZE = 1024  # 버퍼 크기 감소
CHUNK_DURATION = 1.5  # 청크 지속 시간 감소
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.005
audio_queue = queue.Queue(maxsize=50)  # 큐 크기 감소

# 스레드 풀 생성
executor = ThreadPoolExecutor(max_workers=1)  # 워커 수 감소

def preprocess_audio(audio_data):
    """오디오 데이터 전처리 (속도 최적화)"""
    # 간단한 전처리만 수행
    audio_data = audio_data - np.mean(audio_data)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    return audio_data

def is_silent(audio_data):
    """오디오 데이터가 무음인지 확인 (속도 최적화)"""
    return np.sqrt(np.mean(np.square(audio_data))) < SILENCE_THRESHOLD

def audio_callback(indata, frames, time, status):
    """마이크 입력 데이터를 큐에 저장 (속도 최적화)"""
    if status:
        print(f"Status: {status}")
    try:
        audio_queue.put(indata.copy(), block=False)
    except queue.Full:
        pass

def process_audio_chunk(audio_chunk):
    """오디오 청크 처리 (속도 최적화)"""
    try:
        result = model.transcribe(
            audio_chunk,
            language="ko",
            temperature=0.0,  # 결정적 출력
            beam_size=1,      # 빔 서치 크기 감소
            best_of=1,        # 후보 수 감소
            fp16=True,        # FP16 사용
            condition_on_previous_text=False,  # 이전 텍스트 조건 제거
            word_timestamps=False  # 타임스탬프 비활성화
        )
        return result["text"].strip()
    except Exception as e:
        print(f"처리 오류: {str(e)}")
        return ""

def transcribe_audio():
    """큐에서 오디오 데이터를 가져와 실시간으로 텍스트로 변환 (속도 최적화)"""
    print("\n음성 인식 시작...")
    audio_buffer = np.array([], dtype=np.float32)
    last_text = ""
    silence_counter = 0
    processing_future = None

    while True:
        try:
            # 큐에서 데이터 가져오기 (속도 최적화)
            while len(audio_buffer) < CHUNK_SIZE and not audio_queue.empty():
                audio_buffer = np.append(audio_buffer, audio_queue.get_nowait())

            if len(audio_buffer) >= CHUNK_SIZE:
                audio_buffer = preprocess_audio(audio_buffer)
                
                if not is_silent(audio_buffer):
                    silence_counter = 0
                    if processing_future is None or processing_future.done():
                        processing_future = executor.submit(process_audio_chunk, audio_buffer)
                        current_text = processing_future.result()
                        
                        if current_text and current_text != last_text:
                            print(f"인식 결과: {current_text}")
                            last_text = current_text
                else:
                    silence_counter += 1
                    if silence_counter > 2:  # 연속 무음 임계값 감소
                        audio_buffer = np.array([], dtype=np.float32)
                        silence_counter = 0
                
                audio_buffer = np.array([], dtype=np.float32)
            
            time.sleep(0.01)  # 대기 시간 최소화

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            time.sleep(0.1)  # 오류 대기 시간 감소

def main():
    try:
        transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
        transcription_thread.start()

        with sd.InputStream(callback=audio_callback, 
                          channels=1, 
                          samplerate=SAMPLE_RATE, 
                          blocksize=BUFFER_SIZE):
            print("음성을 듣고 있습니다... 종료하려면 Ctrl+C를 누르세요.")
            while True:
                time.sleep(0.01)  # 메인 루프 대기 시간 최소화

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다...")
        executor.shutdown(wait=False)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
