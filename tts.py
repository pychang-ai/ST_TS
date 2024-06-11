import deepspeech
import wave
import numpy as np
from pydub import AudioSegment

# 設置模型路徑
model_file_path = 'deepspeech-0.9.3-models.pbmm'
scorer_file_path = 'deepspeech-0.9.3-models.scorer'

# 加載 DeepSpeech 模型
model = deepspeech.Model(model_file_path)
model.enableExternalScorer(scorer_file_path)

# 將 MP3 文件轉換為 WAV 文件
audio_file_path = 'path/to/your/audio/file.mp3'
sound = AudioSegment.from_mp3(audio_file_path)
wav_file_path = audio_file_path.replace('.mp3', '.wav')
sound.export(wav_file_path, format='wav')

# 讀取 WAV 文件
with wave.open(wav_file_path, 'r') as wf:
    sample_rate = wf.getframerate()
    frames = wf.getnframes()
    buffer = wf.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)

# 使用 DeepSpeech 進行語音轉文字
text = model.stt(data16)
print("Transcription: ", text)
