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
import collections
import queue
import sys
import threading
import time

import numpy as np

from wake_word import is_wake, is_wake_head

SAMPLE_RATE = 16000

# ===== 発話判定(Silero VAD) =====
# faster-whisper に同梱されている Silero VAD(ONNX/CPU)を使う。追加依存はない。
#
# 当初は音量(RMS)で判定していたが、マイクによっては成立しない。内蔵 AGC が
# 無音時にゲインを上げるため、暗騒音が発話と同じ音量まで持ち上がるため。
# 実測(2026-08-04): RMS は 無音の最大 0.0856 と 発話の中央値 0.0411 が重なって
# 分離不能だったのに対し、VAD 確率は 無音 0.044〜0.057 / 発話 0.86〜0.98 と
# 15 倍以上開いた。推論は 1 フレーム 0.13ms(CPU 占有 0.1%)。
CHUNK_SAMPLES = 512                       # Silero の 1 フレーム(32ms @ 16kHz)
VAD_FRAME_CHUNKS = 3                      # まとめて 1 回推論する数
FRAME_SAMPLES = CHUNK_SAMPLES * VAD_FRAME_CHUNKS   # 1536 = 96ms
VAD_SPEECH_PROB = 0.5                     # これ以上なら発話
VAD_SILENCE_PROB = 0.35                   # これ未満なら無音(間は直前の判定を維持=ヒステリシス)

# VAD が「発話」と気づく前のフレームも、さかのぼって発話に含める数。
# VAD の判定は語頭がある程度入ってから立ち上がるため、これが無いと
# 「〜です」の「〜」が欠けて Whisper が拾い損ねる(実機で「漏れる」として現れた)。
PREROLL_FRAMES = 3                        # 96ms × 3 = 288ms さかのぼる

_VAD_LOCK = threading.Lock()              # ONNX セッションは共有なので直列化する


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

# ===== ウェイクワード待受のパラメータ =====
WAKE_SILENCE_HOLD_SEC = 0.5     # 無音がこれだけ続いたら区間の終わり
WAKE_MIN_SPEECH_SEC = 0.35      # これより短い区間は無視(咳払い・クリック音)
WAKE_MAX_SEGMENT_SEC = 3.0      # 判定に使う音声の最大長(ウェイクワードは短いので十分)
WAKE_CHECK_INTERVAL_SEC = 1.0   # 発話が続いていてもこの間隔で判定する
                                # (無音待ちだけだと、間を空けずに喋り続けた時に
                                #  いつまでも判定が走らないため)
WAKE_JOIN_SEC = 2.0             # 直前の区間からこの時間内なら、つないでも判定する
                                # (ゆっくり言うと「やっほー」「クロード」が別区間に割れる)


def rms(block):
    """音声ブロックの実効値。空なら 0.0。

    発話判定には使わない(上のコメント参照)。音量を表示・記録したいときのため残している。
    """
    if block is None or len(block) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


class SileroSpeechDetector:
    """1 フレーム(96ms)が発話かどうかを Silero VAD で判定する。

    音声もモデルもすべてローカル。ネットワークには触れない。
    """

    def __init__(self, speech_prob=VAD_SPEECH_PROB, silence_prob=VAD_SILENCE_PROB):
        self._model = None
        self._speech_prob = speech_prob
        self._silence_prob = silence_prob
        self._in_speech = False
        self.last_prob = 0.0

    def load(self):
        """モデルを先に読み込む(初回だけ ~50ms。音声処理中に引きたくないので明示的に呼ぶ)。"""
        with _VAD_LOCK:
            if self._model is None:
                from faster_whisper.vad import get_vad_model
                self._model = get_vad_model()

    def reset(self):
        self._in_speech = False

    def is_speech(self, frame):
        if self._model is None:
            self.load()
        with _VAD_LOCK:
            prob = float(self._model(frame).max())
        self.last_prob = prob
        # ヒステリシス: 中間の確率では直前の判定を維持し、語中の一瞬の落ち込みで
        # 発話が切れないようにする。
        if prob >= self._speech_prob:
            self._in_speech = True
        elif prob < self._silence_prob:
            self._in_speech = False
        return self._in_speech


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
                 long_speech_sec=None, short_silence_sec=None,
                 queue_size=2, log=None, is_speech=None):
        self._on_segment = on_segment
        self._sr = sample_rate
        # フレームが発話かどうかの判定。既定は Silero VAD。
        # テストでは差し替えて、区切りの状態機械だけを検証する。
        self._detector = None if is_speech else SileroSpeechDetector()
        self._is_speech = is_speech or self._detector.is_speech
        self._pending = np.zeros(0, dtype=np.float32)
        self.last_speech_at = 0.0
        self._silence_hold_sec = silence_hold_sec
        # 発話が long_speech_sec を超えたら、short_silence_sec の息継ぎでも区切る。
        # 長く喋っている間にも前半を確定・貼り付けへ回すため(None なら区切らない)
        self._long_samples = int(long_speech_sec * sample_rate) if long_speech_sec else None
        self._short_silence_sec = short_silence_sec
        self._min_samples = int(min_speech_sec * sample_rate)
        self._max_samples = int(max_segment_sec * sample_rate)
        self._check_interval_sec = check_interval_sec
        self._log = log or (lambda msg: None)
        self._queue = queue.Queue(maxsize=queue_size)
        self._enabled = False
        self._thread = None
        # start()/stop() のたびに進む番号。worker は自分の世代が古くなったら抜ける
        # (確定待ちで止まっている間に終了→再開すると、古い worker が生き残って
        #  新しい worker とキューを取り合い、貼る順番が入れ替わるため)
        self.generation = 0
        self._continuation = False   # 直前を息継ぎで区切った(次の区間はその続き)
        self._gap_sec = 0.0          # 息継ぎで区切ってからの無音の長さ
        self._reset()

    # ---- 状態 ----
    def _reset(self):
        self._preroll = collections.deque(maxlen=PREROLL_FRAMES)
        self._segment = []
        self._segment_samples = 0
        self._voiced_samples = 0
        self._silence_sec = 0.0
        self._since_check_sec = 0.0
        self._in_speech = False

    @property
    def enabled(self):
        return self._enabled

    def preload(self):
        """VAD モデルを先に読み込む(音声が流れ始めてから引かないように)。"""
        if self._detector is not None:
            self._detector.load()

    def current_audio(self):
        """まだ区切られていない(喋っている途中の)区間の音声。無ければ None。

        オーディオコールバックが書き足している最中に別スレッドから読むので、
        リストの参照を一度取ってから複製して使う(_reset は新しいリストに差し替える)。
        """
        if not self._in_speech:
            return None
        blocks = list(self._segment)
        if not blocks:
            return None
        return np.concatenate(blocks, axis=0).flatten()

    def start(self):
        # 直前の worker が終了しかけている場合がある(検出直後など)ので待つ。
        # ここで「生きているから」と早期 return すると _enabled が False のままになり、
        # 二度と発火しない待受になってしまう。
        if self._thread and self._thread.is_alive():
            self._enabled = False
            self._wake_worker()
            self._thread.join(timeout=1.0)
        self._reset()
        self._continuation = False
        self._pending = np.zeros(0, dtype=np.float32)
        if self._detector is not None:
            self._detector.reset()
        # 一度も喋らないまま無音判定されないよう、開始時刻を入れておく
        self.last_speech_at = time.monotonic()
        self._drain()
        self.generation += 1
        self._enabled = True
        self._thread = threading.Thread(target=self._worker, args=(self.generation,), daemon=True)
        self._thread.start()

    def stop(self):
        self._enabled = False
        self.generation += 1
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

    # ---- オーディオコールバックから呼ばれる ----
    # VAD は 1 フレーム 0.13ms(実測)なので、ここで回してもコールバックを詰まらせない。
    def feed(self, block):
        if not self._enabled:
            return
        # Silero は 512 サンプル単位でしか食べないので、固定長フレームに切り直す
        self._pending = np.concatenate([self._pending, block])
        while len(self._pending) >= FRAME_SAMPLES:
            frame = self._pending[:FRAME_SAMPLES]
            self._pending = self._pending[FRAME_SAMPLES:]
            self._feed_frame(frame)

    def _feed_frame(self, block):
        duration = len(block) / self._sr
        if self._is_speech(block):
            if not self._in_speech:
                # 語頭が欠けないよう、VAD が気づく前のフレームもさかのぼって入れる。
                # voiced にはカウントしない(最小発話長の判定を甘くしないため)。
                for earlier in self._preroll:
                    self._append(earlier)
                self._preroll.clear()
            self._in_speech = True
            self._silence_sec = 0.0
            self._voiced_samples += len(block)
            self.last_speech_at = time.monotonic()
            self._append(block)
            if self._check_interval_sec is not None:
                self._since_check_sec += duration
                if self._since_check_sec >= self._check_interval_sec:
                    self._since_check_sec = 0.0
                    self._submit()
        elif self._in_speech:
            self._silence_sec += duration
            hold = self._hold_sec()
            if self._silence_sec <= hold:
                self._append(block)  # 語尾が切れないよう無音の頭までは含める
            if self._silence_sec >= hold:
                breath = hold < self._silence_hold_sec
                gap = self._silence_sec
                self._submit()
                self._reset()
                # 息継ぎで区切ったなら、次の区間は同じ発話の続き(短くても捨てない)
                self._continuation, self._gap_sec = breath, gap
        else:
            self._preroll.append(block)   # 次の発話の語頭用に直前を持っておく
            if self._continuation:
                # 普通の「黙った」長さに達したら、もう続きではない
                self._gap_sec += duration
                if self._gap_sec >= self._silence_hold_sec:
                    self._continuation = False

    def _hold_sec(self):
        """今の区間を区切るのに必要な無音の長さ。長く喋っている間は息継ぎで足りる。"""
        if self._long_samples is not None and self._voiced_samples >= self._long_samples:
            return self._short_silence_sec
        return self._silence_hold_sec

    def _append(self, block):
        if self._segment_samples < self._max_samples:
            self._segment.append(block)
            self._segment_samples += len(block)

    def _submit(self):
        """今の発話区間をキューへ(重い処理は worker が行う)。"""
        # 長さの判定は「音があったフレーム」だけで行う。無音待ち(silence_hold_sec)分を
        # 足した長さで判定すると、咳払いのような一瞬の物音が閾値をすり抜ける。
        # ただし息継ぎで区切った直後の続き(「…です。」の「です」)は短くても通す。
        # 区切る前なら同じ発話に入っていた音なので、ここで捨てると語尾が消える。
        min_samples = 1 if self._continuation else self._min_samples
        if self._voiced_samples < min_samples:
            return
        audio = np.concatenate(self._segment, axis=0).flatten()
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            # 処理が追いつかない。黙って捨てると「喋ったのに貼られない」になるので必ず出す。
            self._log(f"[segmenter] 処理が追いつかず {len(audio) / self._sr:.1f}s を捨てました")

    # ---- 処理スレッド ----
    def _worker(self, generation):
        while self._enabled and generation == self.generation:
            audio = self._queue.get()
            if audio is None or not self._enabled or generation != self.generation:
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

    def __init__(self, transcribe, on_wake, sample_rate=SAMPLE_RATE, log=None,
                 is_speech=None, join_sec=WAKE_JOIN_SEC):
        self._transcribe = transcribe
        self._on_wake = on_wake
        self._log = log or (lambda msg: None)
        self._join_sec = join_sec
        self._prev_text = ""
        self._prev_at = 0.0
        self._segmenter = SpeechSegmenter(
            self._check,
            sample_rate=sample_rate,
            silence_hold_sec=WAKE_SILENCE_HOLD_SEC,
            min_speech_sec=WAKE_MIN_SPEECH_SEC,
            max_segment_sec=WAKE_MAX_SEGMENT_SEC,
            check_interval_sec=WAKE_CHECK_INTERVAL_SEC,
            log=log,
            is_speech=is_speech,
        )

    def preload(self):
        self._segmenter.preload()

    @property
    def enabled(self):
        return self._segmenter.enabled

    def start(self):
        self._prev_text = ""   # 開き直す前の区間は持ち越さない
        self._segmenter.start()

    def stop(self):
        self._segmenter.stop()

    def feed(self, block):
        self._segmenter.feed(block)

    def _check(self, audio):
        text = self._transcribe(audio)
        if not text:
            return
        now = time.monotonic()
        # つなぐのは直前が「やっほー」だけで終わっていたときに限る。何でもつなぐと
        # 「やっほー、元気?」→「クラウドの件」が「やほげんき…くらうど」で当たってしまう
        joined = ""
        if now - self._prev_at <= self._join_sec and is_wake_head(self._prev_text):
            joined = self._prev_text + text
        self._prev_text, self._prev_at = text, now
        hit = is_wake(text) or (bool(joined) and is_wake(joined))
        self._log(f"[wake] {text!r} -> {'HIT' if hit else 'ignore'}")
        if hit:
            self._prev_text = ""
            self._segmenter.stop()  # 二重発火を防ぐ(再開はアプリ側が行う)
            self._on_wake()
