"""TTS生成のtest.wavをlarge-v3で文字起こしし、結果をUTF-8で保存(精度検証用)。"""
import cuda_setup  # noqa: F401  (faster_whisper より前に CUDA DLL パスを登録)
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("test.wav", language="ja", beam_size=5, vad_filter=True)
text = "".join(s.text for s in segments).strip()
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("done")
