"""発話区間の切り出しと、ウェイクワード判定。

マイクストリームは持たない。音声ブロックは `core.VoiceCore` が所有するストリームから
`feed()` に渡される(SECURITY.md「OS に触れる処理は core.py に集約」を守るため)。

- `SpeechSegmenter` … RMS で「喋っている区間」を切り出し、区間ごとに
  `on_segment(audio)` を**別スレッドで**呼ぶ。`feed()` はオーディオコールバックから
  呼ばれるので、重い処理は一切しない(ためてキューに積むだけ)。
- `WakeDetector` … `SpeechSegmenter` の上に載せたウェイクワード判定。

同じ仕組みを 2 通りに使う:
  待受   = 短い区間を切り出して「やっほークロード」かどうかだけ見る(WAKE_* の定数)
  口述   = 喋り終わりの無音で区切って、その区間をまるごと文字起こしに回す(app_rt 側で設定)

音声はメモリ上のバッファのみで、保存も送信もしない。
"""
import queue
import sys
import threading

import numpy as np

from wake_word import is_wake

SAMPLE_RATE = 16000


def safe_log(message):
    """コンソールの文字コードで表現できない文字が来ても落ちないログ出力。

    Windows のコンソールは既定が cp932 で、Whisper は cp932 に無い文字
    (絵文字・稀な漢字など)を返すことがある。素の print だと UnicodeEncodeError が
    飛び、判定スレッドごと死んで「待受は緑なのに何も反応しない」状態になる。
    """
    try:
        print(message)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(message.encode(enc, errors="replace").decode(enc, errors="replace"))
    except Exception:
        pass  # pythonw 起動などで stdout が無い場合。ログのために機能を止めない

# 発話とみなす音量(float32 の RMS)。下げると拾いやすく、上げると誤検知が減る。
SPEECH_RMS = 0.012

# ===== ウェイクワード待受のパラメータ =====
WAKE_SILENCE_HOLD_SEC = 0.5     # 無音がこれだけ続いたら区間の終わり
WAKE_MIN_SPEECH_SEC = 0.35      # これより短い区間は無視(咳払い・クリック音)
WAKE_MAX_SEGMENT_SEC = 3.0      # 判定に使う音声の最大長(ウェイクワードは短いので十分)
WAKE_CHECK_INTERVAL_SEC = 1.0   # 発話が続いていてもこの間隔で判定する
                                # (無音待ちだけだと、間を空けずに喋り続けた時に
                                #  いつまでも判定が走らないため)


def rms(block):
    """音声ブロックの実効値。空なら 0.0。"""
    if block is None or len(block) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


class SpeechSegmenter:
    """音声ブロックを受け取り、発話区間ごとに on_segment(np.ndarray) を呼ぶ。

    on_segment は専用スレッドで**直列に**呼ばれる(文字起こし → 貼り付けの順番が
    入れ替わらないようにするため)。
    """

    def __init__(self, on_segment, *, sample_rate=SAMPLE_RATE,
                 silence_hold_sec=WAKE_SILENCE_HOLD_SEC,
                 min_speech_sec=WAKE_MIN_SPEECH_SEC,
                 max_segment_sec=WAKE_MAX_SEGMENT_SEC,
                 check_interval_sec=None,
                 queue_size=2, log=None):
        self._on_segment = on_segment
        self._sr = sample_rate
        self._silence_hold_sec = silence_hold_sec
        self._min_samples = int(min_speech_sec * sample_rate)
        self._max_samples = int(max_segment_sec * sample_rate)
        self._check_interval_sec = check_interval_sec
        self._log = log or (lambda msg: None)
        self._queue = queue.Queue(maxsize=queue_size)
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

    @property
    def enabled(self):
        return self._enabled

    def start(self):
        # 直前の worker が終了しかけている場合がある(検出直後など)ので待つ。
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
            if self._check_interval_sec is not None:
                self._since_check_sec += duration
                if self._since_check_sec >= self._check_interval_sec:
                    self._since_check_sec = 0.0
                    self._submit()
        elif self._in_speech:
            self._silence_sec += duration
            if self._silence_sec <= self._silence_hold_sec:
                self._append(block)  # 語尾が切れないよう無音の頭までは含める
            if self._silence_sec >= self._silence_hold_sec:
                self._submit()
                self._reset()

    def _append(self, block):
        if self._segment_samples < self._max_samples:
            self._segment.append(block)
            self._segment_samples += len(block)

    def _submit(self):
        """今の発話区間をキューへ(重い処理は worker が行う)。"""
        # 長さの判定は「音があったフレーム」だけで行う。無音待ち(silence_hold_sec)分を
        # 足した長さで判定すると、咳払いのような一瞬の物音が閾値をすり抜ける。
        if self._voiced_samples < self._min_samples:
            return
        audio = np.concatenate(self._segment, axis=0).flatten()
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            # 処理が追いつかない。黙って捨てると「喋ったのに貼られない」になるので必ず出す。
            self._log(f"[segmenter] 処理が追いつかず {len(audio) / self._sr:.1f}s を捨てました")

    # ---- 処理スレッド ----
    def _worker(self):
        while self._enabled:
            audio = self._queue.get()
            if audio is None or not self._enabled:
                continue
            try:
                self._on_segment(audio)
            except Exception as e:  # 1 区間の失敗で待受ごと死なせない
                self._log(f"[segmenter] error: {e}")


class WakeDetector:
    """発話区間ごとにウェイクワードかどうかを判定し、当たれば on_wake() を1回呼ぶ。

    transcribe: np.ndarray -> str   (小さいモデルでの高速文字起こしを想定)
    on_wake:    () -> None
    """

    def __init__(self, transcribe, on_wake, sample_rate=SAMPLE_RATE, log=None):
        self._transcribe = transcribe
        self._on_wake = on_wake
        self._log = log or (lambda msg: None)
        self._segmenter = SpeechSegmenter(
            self._check,
            sample_rate=sample_rate,
            silence_hold_sec=WAKE_SILENCE_HOLD_SEC,
            min_speech_sec=WAKE_MIN_SPEECH_SEC,
            max_segment_sec=WAKE_MAX_SEGMENT_SEC,
            check_interval_sec=WAKE_CHECK_INTERVAL_SEC,
            log=log,
        )

    @property
    def enabled(self):
        return self._segmenter.enabled

    def start(self):
        self._segmenter.start()

    def stop(self):
        self._segmenter.stop()

    def feed(self, block):
        self._segmenter.feed(block)

    def _check(self, audio):
        text = self._transcribe(audio)
        if not text:
            return
        hit = is_wake(text)
        self._log(f"[wake] {text!r} -> {'HIT' if hit else 'ignore'}")
        if hit:
            self._segmenter.stop()  # 二重発火を防ぐ(再開はアプリ側が行う)
            self._on_wake()
