"""GUI 側の配線のテスト(ウェイクワード待受 → 連続口述セッションの受け渡し)。

ウィンドウは作らない。`MicButton` のメソッドをモックの self に対して直接呼び、
「どの順に何を呼ぶか」だけを検証する(Qt のイベントループ・マイク・GPU 不要)。

ここで守りたい不変条件:
- 手押し録音を始める前に、待受側のマイクを閉じる(同じマイクの取り合いを避ける)。
- 待受 → 口述セッションの切り替えでは**マイクを閉じない**(閉じて開き直すと、
  その間の音が落ちて発話の頭が欠ける)。渡し先だけ差し替える。
- セッションを抜けたら待受へ戻る。待受 OFF ならマイクを閉じる(開きっぱなしにしない)。
- 1発話は「文字起こし → 貼り付け」の順で、空文字なら貼らない。
"""
import os
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_rt  # noqa: E402
from app_rt import MicButton  # noqa: E402


def _fake(**attrs):
    fake = mock.MagicMock()
    for k, v in attrs.items():
        setattr(fake, k, v)
    return fake


class ToggleTest(unittest.TestCase):
    def test_stops_listening_before_opening_the_recording_stream(self):
        calls = []
        fake = _fake(state=app_rt.IDLE)
        fake._stop_listening.side_effect = lambda: calls.append("stop_listening")
        fake.core.start_recording.side_effect = lambda: calls.append("start_recording")

        MicButton.toggle(fake)

        self.assertEqual(
            calls[:2], ["stop_listening", "start_recording"],
            "待受のマイクを閉じる前に録音を開こうとしている",
        )

    def test_loading_state_does_nothing(self):
        fake = _fake(state=app_rt.LOADING)
        MicButton.toggle(fake)
        fake.core.start_recording.assert_not_called()

    def test_toggle_during_session_ends_it(self):
        # 口述セッション中にボタン/ホットキーが来たら「止める」操作になる
        fake = _fake(state=app_rt.SESSION)
        MicButton.toggle(fake)
        fake._end_session.assert_called_once()
        fake.core.start_recording.assert_not_called()


class SessionTest(unittest.TestCase):
    def test_starting_a_session_swaps_the_callback_without_closing_the_mic(self):
        fake = _fake(wake_on=True)
        with mock.patch.object(app_rt.threading, "Thread"):
            MicButton._start_session(fake)

        fake._wake.stop.assert_called_once()
        fake._dictate.start.assert_called_once()
        fake.core.start_monitor.assert_called_once_with(fake._dictate.feed)
        fake.core.stop_monitor.assert_not_called()   # ここで閉じると発話の頭が欠ける
        fake.state_changed.emit.assert_called_once_with(app_rt.SESSION)

    def test_session_shows_and_hides_the_preview_panel(self):
        # 何が入力されるのか見えないと、拾えているのか判断できない
        fake = _fake(wake_on=True)
        with mock.patch.object(app_rt.threading, "Thread"):
            MicButton._start_session(fake)
        fake._show_preview.assert_called_once()

        fake2 = _fake(wake_on=True)
        MicButton._end_session(fake2)
        fake2.preview.hide.assert_called_once()

    def test_each_utterance_is_shown_before_being_pasted(self):
        fake = _fake()
        fake.core.transcribe.return_value = "こんにちは"
        MicButton._on_utterance(fake, np.zeros(16000, dtype=np.float32))
        fake.preview_text.emit.assert_called_once_with("こんにちは")

    def test_unrecognized_utterance_says_so_instead_of_staying_blank(self):
        fake = _fake()
        fake.core.transcribe.return_value = ""
        MicButton._on_utterance(fake, np.zeros(16000, dtype=np.float32))
        fake.preview_text.emit.assert_called_once_with("(認識できませんでした)")
        fake.core.deliver.assert_not_called()

    def test_starting_a_session_launches_the_idle_watchdog(self):
        fake = _fake(wake_on=True)
        with mock.patch.object(app_rt.threading, "Thread") as thread:
            MicButton._start_session(fake)
        self.assertIs(thread.call_args.kwargs["target"], fake._session_watchdog)

    def test_ending_a_session_returns_to_listening_when_wake_is_on(self):
        fake = _fake(wake_on=True)
        MicButton._end_session(fake)
        fake._dictate.stop.assert_called_once()
        fake.core.stop_monitor.assert_not_called()   # 待受が続くのでマイクは開いたまま
        fake.state_changed.emit.assert_called_once_with(app_rt.IDLE)

    def test_ending_a_session_closes_the_mic_when_wake_is_off(self):
        fake = _fake(wake_on=False)
        MicButton._end_session(fake)
        fake.core.stop_monitor.assert_called_once()  # 開きっぱなしにしない

    def test_turning_wake_off_during_a_session_ends_it(self):
        fake = _fake(state=app_rt.SESSION)
        MicButton.set_wake_mode(fake, False)
        fake._end_session.assert_called_once()
        fake._stop_listening.assert_not_called()

    def test_watchdog_ends_the_session_after_a_long_silence(self):
        fake = _fake(state=app_rt.SESSION)
        fake._dictate.last_speech_at = time.monotonic() - app_rt.SESSION_IDLE_TIMEOUT_SEC - 1
        with mock.patch.object(app_rt.time, "sleep"):
            MicButton._session_watchdog(fake)
        fake.notify.emit.assert_called_once_with("__end_session__")

    def test_watchdog_keeps_the_session_while_speaking(self):
        # 発話時刻は segmenter が VAD の判定で更新する。音量では更新しない
        # (暗騒音が大きいマイクだと、音量基準では永遠にタイムアウトしない)
        fake = _fake(state=app_rt.SESSION)
        fake._dictate.last_speech_at = time.monotonic()
        ticks = []

        def sleep(_):
            ticks.append(1)
            if len(ticks) >= 3:
                fake.state = app_rt.IDLE   # セッションが別経路で終わった相当

        with mock.patch.object(app_rt.time, "sleep", side_effect=sleep):
            MicButton._session_watchdog(fake)
        fake.notify.emit.assert_not_called()


class UtteranceTest(unittest.TestCase):
    def test_transcribes_then_delivers(self):
        calls = []
        fake = _fake()
        fake.core.transcribe.side_effect = lambda a: (calls.append("transcribe"), "こんにちは")[1]
        fake.core.deliver.side_effect = lambda t: calls.append("deliver")

        MicButton._on_utterance(fake, np.zeros(16000, dtype=np.float32))

        self.assertEqual(calls, ["transcribe", "deliver"])
        fake.core.deliver.assert_called_once_with("こんにちは")

    def test_empty_transcription_is_not_pasted(self):
        fake = _fake()
        fake.core.transcribe.return_value = ""
        MicButton._on_utterance(fake, np.zeros(16000, dtype=np.float32))
        fake.core.deliver.assert_not_called()


class ListeningLifecycleTest(unittest.TestCase):
    def test_returning_to_idle_restarts_listening(self):
        fake = _fake(wake_on=True)
        MicButton._on_state(fake, app_rt.IDLE)
        fake._start_listening.assert_called_once()

    def test_does_not_restart_while_busy(self):
        fake = _fake(wake_on=True)
        MicButton._on_state(fake, app_rt.BUSY)
        fake._start_listening.assert_not_called()

    def test_does_not_restart_when_wake_mode_is_off(self):
        fake = _fake(wake_on=False)
        MicButton._on_state(fake, app_rt.IDLE)
        fake._start_listening.assert_not_called()

    def test_start_listening_is_a_noop_while_in_a_session(self):
        fake = _fake(wake_on=True, state=app_rt.SESSION)
        MicButton._start_listening(fake)
        fake.core.start_monitor.assert_not_called()

    def test_start_listening_opens_the_monitor_when_idle(self):
        fake = _fake(wake_on=True, state=app_rt.IDLE)
        MicButton._start_listening(fake)
        fake._wake.start.assert_called_once()
        fake.core.start_monitor.assert_called_once_with(fake._wake.feed)

    def test_start_listening_turns_wake_off_if_the_mic_cannot_be_opened(self):
        # マイクが他アプリに掴まれている等で開けなかったとき、
        # 「緑なのに聞いていない」状態にならないこと
        fake = _fake(wake_on=True, state=app_rt.IDLE)
        fake.core.start_monitor.side_effect = OSError("device busy")
        MicButton._start_listening(fake)
        self.assertFalse(fake.wake_on)
        fake._wake.stop.assert_called_once()


class NotifyTest(unittest.TestCase):
    def test_wake_notification_starts_a_session(self):
        fake = _fake(state=app_rt.IDLE)
        MicButton._on_notify(fake, "__wake__")
        fake._start_session.assert_called_once()

    def test_wake_notification_while_recording_returns_to_listening(self):
        fake = _fake(state=app_rt.RECORDING)
        MicButton._on_notify(fake, "__wake__")
        fake._start_session.assert_not_called()
        fake._start_listening.assert_called_once()

    def test_end_session_notification(self):
        fake = _fake(state=app_rt.SESSION)
        MicButton._on_notify(fake, "__end_session__")
        fake._end_session.assert_called_once()

    def test_end_session_notification_is_ignored_when_not_in_a_session(self):
        fake = _fake(state=app_rt.IDLE)
        MicButton._on_notify(fake, "__end_session__")
        fake._end_session.assert_not_called()

    def test_hotkey_toggles_wake_mode(self):
        fake = _fake(wake_on=False)
        MicButton._on_notify(fake, "__wake_toggle__")
        fake.set_wake_mode.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
