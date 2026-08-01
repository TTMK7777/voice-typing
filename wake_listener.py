"""ウェイクワード待受のロジック層(発話区間の切り出し + ウェイク判定)。

マイクストリームは持たない。音声ブロックは `core.VoiceCore` が所有するストリームから
`feed()` に渡される(SECURITY.md「OS に触れる処理は core.py に集約」を守るため)。

流れ:
  feed()  … オーディオコールバックから呼ばれる。RMS で発話中かを判定し、
            発話区間をためて「判定キュー」に積むだけ(重い処理は一切しない)。
  worker  … 別スレッドでキューを取り出し、small モデルで文字起こし →
            ウェイクワードなら on_wake() を1回だけ呼ぶ。

判定に使った音声はメモリ上のバッファのみで、保存も送信もしない。
"""
import queue
import threading

import numpy as np

from wake_word import is_wake

SAMPLE_RATE = 16000

# ===== チューニング用パラメータ =====
SPEECH_RMS = 0.012          # これ以上の音量を「発話中」とみなす(float32 の RMS)
SILENCE_HOLD_SEC = 0.5      # 無音がこれだけ続いたら発話区間の終わりとする
MIN_SPEECH_SEC = 0.35       # これより短い区間は無視する(咳払い・クリック音)
MAX_SEGMENT_SEC = 3.0       # ウェイク判定に使う音声の最大長(先頭から)
CHECK_INTERVAL_SEC = 1.0    # 発話が続いていてもこの間隔で判定する
                            # (無音を待つ設計だけだと、間を空けずに喋り続けた時に
                            #  ウェイクワードの判定がいつまでも走らないため)


def rms(block):
    """音声ブロックの実効値。空なら 0.0。"""
    if block is None or len(block) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


class WakeDetector:
    """発話区間を切り出してウェイクワードを判定する。

    transcribe: np.ndarray -> str   (小さいモデルでの高速文字起こしを想定)
    on_wake:    () -> None          (ウェイクワード検出時に1回だけ呼ばれる)
    """

    def __init__(self, transcribe, on_wake, sample_rate=SAMPLE_RATE, log=None):
        self._transcribe = transcribe
        self._on_wake = on_wake
        self._sr = sample_rate
        self._log = log or (lambda msg: None)
        self._max_samples = int(MAX_SEGMENT_SEC * sample_rate)
        self._min_samples = int(MIN_SPEECH_SEC * sample_rate)
        self._queue = queue.Queue(maxsize=2)
        self._enabled = False
        self._thread = None
        self._reset()

    # ---- 状態 ----
    def _reset(self):
        self._segment = []
        self._segment_samples = 0
        self._voiced_samples = 0
        self._silence_sec = 0.0
        self._since_check_sec = 0.0
        self._in_speech = False

    def start(self):
        # 直前の worker が終了しかけている場合がある(ウェイク検出直後など)ので待つ。
        # ここで「生きているから」と早期 return すると _enabled が False のままになり、
        # 二度と発火しない待受になってしまう。
        if self._thread and self._thread.is_alive():
            self._enabled = False
            self._wake_worker()
            self._thread.join(timeout=1.0)
        self._reset()
        self._drain()
        self._enabled = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._enabled = False
        self._reset()
        self._wake_worker()

    def _wake_worker(self):
        """キュー待ちで止まっている worker を起こす(番兵を入れる)。"""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass  # 満杯 = worker は待っていないので起こす必要がない

    def _drain(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # ---- オーディオコールバックから呼ばれる(軽い処理のみ) ----
    def feed(self, block):
        if not self._enabled:
            return
        duration = len(block) / self._sr
        if rms(block) >= SPEECH_RMS:
            self._in_speech = True
            self._silence_sec = 0.0
            self._voiced_samples += len(block)
            self._append(block)
            self._since_check_sec += duration
            if self._since_check_sec >= CHECK_INTERVAL_SEC:
                self._since_check_sec = 0.0
                self._submit()
        elif self._in_speech:
            self._silence_sec += duration
            if self._silence_sec <= SILENCE_HOLD_SEC:
                self._append(block)  # 語尾が切れないよう無音の頭までは含める
            if self._silence_sec >= SILENCE_HOLD_SEC:
                self._submit()
                self._reset()

    def _append(self, block):
        if self._segment_samples < self._max_samples:
            self._segment.append(block)
            self._segment_samples += len(block)

    def _submit(self):
        """今の発話区間を判定キューへ(判定自体は worker が行う)。"""
        # 長さの判定は「音があったフレーム」だけで行う。無音待ち(SILENCE_HOLD_SEC)分を
        # 足した長さで判定すると、咳払いのような一瞬の物音が閾値をすり抜ける。
        if self._voiced_samples < self._min_samples:
            return
        audio = np.concatenate(self._segment, axis=0).flatten()
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            pass  # 判定が追いつかないときは捨てる(遅れて発火するより落とす方がまし)

    # ---- 判定スレッド ----
    def _worker(self):
        while self._enabled:
            audio = self._queue.get()
            if audio is None or not self._enabled:
                continue
            try:
                text = self._transcribe(audio)
            except Exception as e:  # 文字起こしが転んでも待受は続ける
                self._log(f"[wake] transcribe error: {e}")
                continue
            if not text:
                continue
            hit = is_wake(text)
            self._log(f"[wake] {text!r} -> {'HIT' if hit else 'ignore'}")
            if hit:
                self._enabled = False  # 二重発火を防ぐ(再開はアプリ側が行う)
                self._on_wake()
