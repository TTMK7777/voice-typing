"""GUI 側の配線のテスト(ウェイクワード待受と録音のマイク受け渡し)。

ウィンドウは作らない。`MicButton` のメソッドをモックの self に対して直接呼び、
「どの順に何を呼ぶか」だけを検証する(Qt のイベントループ・マイク・GPU 不要)。

ここで守りたい不変条件:
- 録音を始める前に、必ず待受側のマイクを閉じる(同じマイクを2つのストリームで
  取り合うと、録音が開けない/無音になる)。
- 録音が終わって IDLE に戻ったら、待受 ON なら待受を再開する
  (再開しないと「1回喋ったきり反応しなくなる」)。
- 録音中は待受を開き直さない。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_rt  # noqa: E402
from app_rt import MicButton  # noqa: E402


def _fake(**attrs):
    fake = mock.Mock()
    for k, v in attrs.items():
        setattr(fake, k, v)
    return fake


class ToggleOrderingTest(unittest.TestCase):
    def test_stops_listening_before_opening_the_recording_stream(self):
        calls = []
        fake = _fake(state=app_rt.IDLE, _wake_session=False)
        fake._stop_listening.side_effect = lambda: calls.append("stop_listening")
        fake.core.start_recording.side_effect = lambda: calls.append("start_recording")

        MicButton.toggle(fake)

        self.assertEqual(
            calls[:2], ["stop_listening", "start_recording"],
            "待受のマイクを閉じる前に録音を開こうとしている",
        )

    def test_loading_state_does_nothing(self):
        fake = _fake(state=app_rt.LOADING, _wake_session=False)
        MicButton.toggle(fake)
        fake.core.start_recording.assert_not_called()

    def test_wake_session_starts_the_silence_watchdog(self):
        fake = _fake(state=app_rt.IDLE, _wake_session=True)
        with mock.patch.object(app_rt.threading, "Thread") as thread:
            MicButton.toggle(fake)
        thread.assert_called_once()
        self.assertIs(thread.call_args.kwargs["target"], fake._silence_watchdog)

    def test_manual_start_has_no_watchdog(self):
        # 手で押して始めた録音は、自分で止めるまで止まらない(勝手に確定しない)
        fake = _fake(state=app_rt.IDLE, _wake_session=False)
        with mock.patch.object(app_rt.threading, "Thread") as thread:
            MicButton.toggle(fake)
        thread.assert_not_called()


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

    def test_start_listening_is_a_noop_while_recording(self):
        fake = _fake(wake_on=True, state=app_rt.RECORDING)
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
    def test_wake_notification_starts_a_wake_session(self):
        fake = _fake(state=app_rt.IDLE)
        MicButton._on_notify(fake, "__wake__")
        self.assertTrue(fake._wake_session)
        fake.toggle.assert_called_once()

    def test_wake_notification_while_recording_returns_to_listening(self):
        fake = _fake(state=app_rt.RECORDING)
        MicButton._on_notify(fake, "__wake__")
        fake.toggle.assert_not_called()
        fake._start_listening.assert_called_once()

    def test_hotkey_toggles_wake_mode(self):
        fake = _fake(wake_on=False)
        MicButton._on_notify(fake, "__wake_toggle__")
        fake.set_wake_mode.assert_called_once_with(True)


class FinishTest(unittest.TestCase):
    def test_wake_session_is_cleared_even_if_transcription_raises(self):
        fake = _fake(_wake_session=True)
        fake.core.stop_recording.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            MicButton._finish(fake)
        self.assertFalse(fake._wake_session, "例外時にウェイクセッションが残っている")


if __name__ == "__main__":
    unittest.main()
