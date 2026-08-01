"""ウェイクワード「やっほー Claude」の判定と、発話区間の切り出しのテスト。

追加依存ゼロ(stdlib unittest + numpy)。実行:
    .venv/Scripts/python.exe -m unittest discover -s tests -v

検証の要点:
- Whisper の表記揺れ(カタカナ/ひらがな/長音/句読点/ローマ字)を吸収して検出できる。
- 似ているが違う発話(「クロード、クリア」等)で誤発火しない。
- WakeDetector が無音では判定を投げず、発話区間だけを判定に回す。

GPU / マイク不要(文字起こしはダミー関数に差し替える)。
"""
import os
import sys
import time
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wake_listener  # noqa: E402
from wake_word import is_wake, normalize  # noqa: E402


class NormalizeTest(unittest.TestCase):
    def test_katakana_becomes_hiragana(self):
        self.assertEqual(normalize("クロード"), "くろど")

    def test_drops_prolonged_and_small_tsu(self):
        self.assertEqual(normalize("ヤッホー"), "やほ")
        self.assertEqual(normalize("やっほー"), "やほ")
        self.assertEqual(normalize("やほー"), "やほ")

    def test_drops_punctuation_and_spaces(self):
        self.assertEqual(normalize("ヤッホー、クロード。"), "やほくろど")
        self.assertEqual(normalize("ヤッホー　クロード"), "やほくろど")

    def test_fullwidth_alnum_becomes_halfwidth_lower(self):
        self.assertEqual(normalize("Ｃｌａｕｄｅ"), "claude")

    def test_none_and_empty(self):
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize(""), "")


class IsWakeTest(unittest.TestCase):
    def test_accepts_transcription_variants(self):
        # Whisper が同じ発話に対して返しうる表記のバリエーション
        for text in [
            "ヤッホー、クロード",
            "やっほークロード",
            "やっほー、クロード。",
            "ヤッホー クロード",
            "やほークロード",
            "ヤッホークラウド",
            "やっほー、クラウド!",
            "ヤッホウ、クロード",
            "やっほーclaude",
            "Yahoo Claude",
            "ヤッホー、クロート",
            "あ、やっほークロード",
            # 以下は 2026-08-01 の実機ログで実際に Whisper が返した表記
            "やほうくろうどう",
            "やっほーくろーどう",
            "やっほーくろーどー",
            "やほくらどう",
        ]:
            with self.subTest(text=text):
                self.assertTrue(is_wake(text), f"{text!r} を検出できていない")

    def test_rejects_non_wake_speech(self):
        for text in [
            "クロード、クリア",           # 音声コマンド(ウェイクワードではない)
            "クロード、コンパクト",
            "やっほー、元気ですか",        # 「やっほー」だけ
            "やはりクロード",              # 実機ログで拾った空似(2026-08-01)
            "おはようございます",
            "今日はクロードと話した",      # クロードだけ + 文中
            "クラウドの設定を確認する",
            "ご視聴ありがとうございました",  # Whisper の定番幻覚
            "",
            None,
        ]:
            with self.subTest(text=text):
                self.assertFalse(is_wake(text), f"{text!r} で誤発火している")

    def test_rejects_wake_word_in_the_middle(self):
        # 文の途中に出てきた場合は起動しない(先頭 3 文字までの前置きのみ許容)
        self.assertFalse(is_wake("さっき彼にやっほークロードって言ったんだよ"))


def _tone(seconds, amplitude, sample_rate=wake_listener.SAMPLE_RATE):
    """発話の代わりに使う一定振幅の信号。RMS = amplitude / sqrt(2)。"""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * 440 * t / sample_rate)).astype(np.float32)


def _silence(seconds, sample_rate=wake_listener.SAMPLE_RATE):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


class RmsTest(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(wake_listener.rms(None), 0.0)
        self.assertEqual(wake_listener.rms(np.array([], dtype=np.float32)), 0.0)

    def test_silence_is_below_threshold(self):
        self.assertLess(wake_listener.rms(_silence(0.1)), wake_listener.SPEECH_RMS)

    def test_loud_tone_is_above_threshold(self):
        self.assertGreater(wake_listener.rms(_tone(0.1, 0.2)), wake_listener.SPEECH_RMS)


class WakeDetectorTest(unittest.TestCase):
    def setUp(self):
        self.seen = []       # 判定に回された音声の長さ(秒)
        self.fired = []
        self.reply = ""

    def _make(self):
        def transcribe(audio):
            self.seen.append(len(audio) / wake_listener.SAMPLE_RATE)
            return self.reply

        return wake_listener.WakeDetector(
            transcribe=transcribe,
            on_wake=lambda: self.fired.append(True),
        )

    def _feed(self, det, signal, block_sec=0.1):
        step = int(block_sec * wake_listener.SAMPLE_RATE)
        for i in range(0, len(signal), step):
            det.feed(signal[i:i + step])

    def _wait(self, det, timeout=2.0):
        """worker の処理が落ち着く(呼び出し回数が変化しなくなる)まで待つ。

        キューが空になるのを待つ方式は使えない: stop() は worker を起こすための
        番兵をキューに入れるため、検出後は「空にならない」まま待ち続けてしまう。
        """
        deadline = time.time() + timeout
        stable = 0
        last = None
        while time.time() < deadline:
            snapshot = (len(self.seen), len(self.fired))
            stable = stable + 1 if snapshot == last else 0
            if stable >= 2:
                return
            last = snapshot
            time.sleep(0.03)

    def test_silence_never_reaches_transcribe(self):
        det = self._make()
        det.start()
        self._feed(det, _silence(3.0))
        self._wait(det)
        det.stop()
        self.assertEqual(self.seen, [], "無音なのに文字起こしが走っている")

    def test_short_blip_is_ignored(self):
        det = self._make()
        det.start()
        # 0.2 秒だけの物音 → MIN_SPEECH_SEC(0.35) 未満なので捨てられる
        self._feed(det, np.concatenate([_tone(0.2, 0.2), _silence(1.0)]))
        self._wait(det)
        det.stop()
        self.assertEqual(self.seen, [])

    def test_speech_segment_is_submitted_after_silence(self):
        det = self._make()
        det.start()
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        det.stop()
        self.assertTrue(self.seen, "発話区間が判定に回っていない")
        self.assertGreaterEqual(self.seen[0], 0.8)

    def test_fires_on_wake_word(self):
        det = self._make()
        self.reply = "ヤッホー、クロード"
        det.start()
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        self.assertEqual(len(self.fired), 1)
        det.stop()

    def test_does_not_fire_on_other_speech(self):
        det = self._make()
        self.reply = "今日はいい天気ですね"
        det.start()
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        det.stop()
        self.assertEqual(self.fired, [])

    def test_disables_itself_after_firing(self):
        # 検出後は自分で待受を止める(アプリが録音へ移る間に二重発火しないため)
        det = self._make()
        self.reply = "やっほークロード"
        det.start()
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        self.assertFalse(det.enabled)
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        det.stop()
        self.assertEqual(len(self.fired), 1, "検出後にもう一度発火している")

    def test_restart_after_firing_works(self):
        # 録音が終わって待受へ戻ったとき、ちゃんともう一度発火できること
        det = self._make()
        self.reply = "やっほークロード"
        det.start()
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        det.start()   # アプリが IDLE に戻って待受を再開する相当
        self.assertTrue(det.enabled)
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        det.stop()
        self.assertEqual(len(self.fired), 2, "待受を再開しても発火しない")

    def test_segment_is_capped(self):
        det = self._make()
        det.start()
        # 10 秒喋り続けても、判定に回すのは MAX_SEGMENT_SEC まで
        self._feed(det, np.concatenate([_tone(10.0, 0.2), _silence(1.0)]))
        self._wait(det)
        det.stop()
        self.assertTrue(self.seen)
        for length in self.seen:
            self.assertLessEqual(length, wake_listener.WAKE_MAX_SEGMENT_SEC + 0.15)

    def test_long_speech_is_checked_without_waiting_for_silence(self):
        # 間を空けずに喋り続けても、CHECK_INTERVAL_SEC ごとに判定が走る
        det = self._make()
        det.start()
        self._feed(det, _tone(3.0, 0.2))
        self._wait(det)
        det.stop()
        self.assertTrue(self.seen, "無音待ちだけだと長い発話で判定が走らない")

    def test_feed_before_start_is_ignored(self):
        det = self._make()
        self._feed(det, np.concatenate([_tone(0.8, 0.2), _silence(1.0)]))
        self._wait(det)
        self.assertEqual(self.seen, [])


class _Cp932Stdout:
    """cp932 のコンソールを模したダミー stdout(表現できない文字で例外を投げる)。"""

    encoding = "cp932"

    def __init__(self):
        self.written = []

    def write(self, s):
        s.encode("cp932")   # 表現できなければ UnicodeEncodeError
        self.written.append(s)

    def flush(self):
        pass


class SafeLogTest(unittest.TestCase):
    """ログ出力でアプリを死なせないこと。

    Windows のコンソールは既定 cp932 で、Whisper は cp932 に無い文字を返しうる。
    素の print だと UnicodeEncodeError で判定スレッドごと落ち、
    「緑なのに何も反応しない」状態になる。
    """

    def test_plain_text_is_printed(self):
        out = _Cp932Stdout()
        with unittest.mock.patch.object(sys, "stdout", out):
            wake_listener.safe_log("[wake] 待受 ON")
        self.assertIn("[wake] 待受 ON", "".join(out.written))

    def test_unencodable_text_does_not_raise(self):
        out = _Cp932Stdout()
        with unittest.mock.patch.object(sys, "stdout", out):
            wake_listener.safe_log("[wake] 'やっほークロード🙂—' -> HIT")
        self.assertTrue(out.written, "置換して出力されていない")

    def test_detection_survives_unencodable_transcription(self):
        # ログが落ちても検出は成立すること(ログのために機能を止めない)
        fired = []
        out = _Cp932Stdout()
        det = wake_listener.WakeDetector(
            transcribe=lambda audio: "やっほークロード🙂",
            on_wake=lambda: fired.append(True),
            log=wake_listener.safe_log,
        )
        with unittest.mock.patch.object(sys, "stdout", out):
            det.start()
            step = int(0.1 * wake_listener.SAMPLE_RATE)
            signal = np.concatenate([_tone(0.8, 0.2), _silence(1.0)])
            for i in range(0, len(signal), step):
                det.feed(signal[i:i + step])
            deadline = time.time() + 2.0
            while time.time() < deadline and not fired:
                time.sleep(0.02)
            det.stop()
        self.assertEqual(len(fired), 1, "ログの文字コードで検出ごと死んでいる")


class DictationSegmenterTest(unittest.TestCase):
    """連続口述モードの設定(黙るまでが1発話 / 途中で切らない)の検証。"""

    def setUp(self):
        self.segments = []

    def _make(self, **kw):
        opts = dict(
            silence_hold_sec=0.7,
            min_speech_sec=0.5,
            max_segment_sec=60.0,
            check_interval_sec=None,   # 途中で区切らない
            queue_size=8,
        )
        opts.update(kw)
        return wake_listener.SpeechSegmenter(
            lambda audio: self.segments.append(len(audio) / wake_listener.SAMPLE_RATE),
            **opts,
        )

    def _feed(self, seg, signal, block_sec=0.1):
        step = int(block_sec * wake_listener.SAMPLE_RATE)
        for i in range(0, len(signal), step):
            seg.feed(signal[i:i + step])

    def _wait(self, timeout=2.0):
        deadline = time.time() + timeout
        stable, last = 0, None
        while time.time() < deadline:
            stable = stable + 1 if len(self.segments) == last else 0
            if stable >= 2:
                return
            last = len(self.segments)
            time.sleep(0.03)

    def test_each_pause_produces_one_segment(self):
        # 喋る → 黙る → 喋る → 黙る で 2 回貼られること
        seg = self._make()
        seg.start()
        self._feed(seg, np.concatenate([
            _tone(1.0, 0.2), _silence(1.0),
            _tone(1.0, 0.2), _silence(1.0),
        ]))
        self._wait()
        seg.stop()
        self.assertEqual(len(self.segments), 2, "無音ごとに1発話にならない")

    def test_short_pause_does_not_split(self):
        # 0.3 秒の「間」では切らない(silence_hold_sec=0.7 未満なので同じ発話)
        seg = self._make()
        seg.start()
        self._feed(seg, np.concatenate([
            _tone(1.0, 0.2), _silence(0.3), _tone(1.0, 0.2), _silence(1.2),
        ]))
        self._wait()
        seg.stop()
        self.assertEqual(len(self.segments), 1, "文中の短い間で切れている")

    def test_long_speech_is_not_chopped_mid_sentence(self):
        # check_interval_sec=None なので、黙るまでは何秒喋っても1発話のまま
        seg = self._make()
        seg.start()
        self._feed(seg, _tone(6.0, 0.2))
        self._wait(timeout=1.0)
        seg.stop()
        self.assertEqual(self.segments, [], "黙っていないのに途中で貼られている")

    def test_backlog_is_reported_not_silently_dropped(self):
        # 文字起こしが詰まって捨てるときは必ずログに出す(黙って消えるのが最悪)
        logs = []
        seg = self._make(queue_size=1, log=logs.append)
        seg._enabled = True          # worker は動かさずキューだけ埋める
        for _ in range(3):
            self._feed(seg, np.concatenate([_tone(1.0, 0.2), _silence(1.0)]))
        self.assertTrue(any("捨てました" in m for m in logs), "取りこぼしが無言で消えている")


if __name__ == "__main__":
    unittest.main()
