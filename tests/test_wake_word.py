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
        ]:
            with self.subTest(text=text):
                self.assertTrue(is_wake(text), f"{text!r} を検出できていない")

    def test_rejects_non_wake_speech(self):
        for text in [
            "クロード、クリア",           # 音声コマンド(ウェイクワードではない)
            "クロード、コンパクト",
            "やっほー、元気ですか",        # 「やっほー」だけ
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
        """worker が判定を終えるまで待つ。"""
        deadline = time.time() + timeout
        while time.time() < deadline and not det._queue.empty():
            time.sleep(0.01)
        time.sleep(0.05)

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
        self.assertFalse(det._enabled)
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
        self.assertTrue(det._enabled)
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
            self.assertLessEqual(length, wake_listener.MAX_SEGMENT_SEC + 0.15)

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


if __name__ == "__main__":
    unittest.main()
