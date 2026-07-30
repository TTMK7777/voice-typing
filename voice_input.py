#!/usr/bin/env python3
"""
voice-typing: ローカル Whisper によるグローバル音声入力

ホットキーでトグル録音 → faster-whisper(GPU)で文字起こし → 今カーソルがある所に貼り付け。
どのアプリでも使える(クリップボード経由で Ctrl+V を送出)。

使い方:
    .venv/Scripts/python.exe voice_input.py
    ホットキー(既定 Ctrl+Alt+Space)で 録音開始 → もう一度押すと停止して入力。
    Ctrl+C で終了。
"""
import time
import threading

import cuda_setup  # noqa: F401  (faster_whisper より前に CUDA DLL パスを登録)
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard

# ===== 設定 =====
MODEL_SIZE = "large-v3"          # 精度最優先。GPU 12GB なら余裕
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "ja"
SAMPLE_RATE = 16000
HOTKEY = "<ctrl>+<alt>+<space>"  # 録音 開始/停止 トグル
VOCAB_FILE = "vocab.txt"         # 固有名詞・専門用語のヒント(1行1語、任意)
RESTORE_CLIPBOARD = True         # 入力後に元のクリップボード内容を復元するか

# ===== 状態 =====
recording = False
frames = []
stream = None
lock = threading.Lock()
kb = keyboard.Controller()


def load_vocab():
    """vocab.txt があれば initial_prompt として読み込む(認識のヒント)。"""
    try:
        with open(VOCAB_FILE, encoding="utf-8") as f:
            words = [w.strip() for w in f if w.strip()]
        return "、".join(words) if words else None
    except FileNotFoundError:
        return None


def audio_callback(indata, frame_count, time_info, status):
    if recording:
        frames.append(indata.copy())


def start_recording():
    global recording, frames, stream
    frames = []
    recording = True
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback
    )
    stream.start()
    print("● 録音中... (もう一度ホットキーで停止)")


def stop_and_transcribe():
    """録音を止め、別スレッドで文字起こし→貼り付け(ホットキー応答をブロックしない)。"""
    global recording, stream
    recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None
    if not frames:
        print("(音声なし)")
        return
    audio = np.concatenate(frames, axis=0).flatten()
    threading.Thread(target=_transcribe_and_paste, args=(audio,), daemon=True).start()


def _transcribe_and_paste(audio):
    print("変換中...")
    t0 = time.time()
    segments, _ = model.transcribe(
        audio,
        language=LANGUAGE,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        beam_size=5,
    )
    text = "".join(seg.text for seg in segments).strip()
    dt = time.time() - t0
    if not text:
        print("(認識結果なし)")
        return
    print(f"認識 ({dt:.1f}s): {text}")
    paste(text)


def paste(text):
    """クリップボード経由でカーソル位置に貼り付け。"""
    old = None
    if RESTORE_CLIPBOARD:
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    kb.press(keyboard.Key.ctrl)
    kb.press("v")
    kb.release("v")
    kb.release(keyboard.Key.ctrl)
    time.sleep(0.15)
    if RESTORE_CLIPBOARD and old is not None:
        pyperclip.copy(old)


def toggle():
    with lock:
        if not recording:
            start_recording()
        else:
            stop_and_transcribe()


def main():
    print("Whisper モデルをロード中... (初回はDLで数分かかります)")
    global model, INITIAL_PROMPT
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    INITIAL_PROMPT = load_vocab()
    print(f"準備完了。 {HOTKEY} で録音 開始/停止。 Ctrl+C で終了。")
    if INITIAL_PROMPT:
        print(f"語彙ヒント読込: {INITIAL_PROMPT[:50]}...")

    with keyboard.GlobalHotKeys({HOTKEY: toggle}) as h:
        h.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了")
