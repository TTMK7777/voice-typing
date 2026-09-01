"""GUI 側の配線のテスト(ウェイクワード待受 → 連続口述セッションの受け渡し)。

ウィンドウは作らない。`MicButton` のメソッドをモックの self に対して直接呼び、
「どの順に何を呼ぶか」だけを検証する(Qt のイベントループ・マイク・GPU 不要)。

ここで守りたい不変条件:
- 手押し録音を始める前に、待受側のマイクを閉じる(同じマイクの取り合いを避ける)。
- 待受 → 口述セッションの切り替えでは**マイクを閉じない**(閉じて開き直すと、
  その間の音が落ちて発話の頭が欠ける)。渡し先だけ差し替える。
- セッションを抜けたら待受へ戻る。待受 OFF ならマイクを閉じる(開きっぱなしにしない)。
- 1発話は「文字起こし → 貼り付け」の順で、空文字なら貼らない。
- 起動直後(モデルロード中)でも録音は始められ、確定はモデルが揃うまで待つ。
  ロード完了は録音中の状態を上書きしない。
"""
import os
import sys
import threading
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

    def test_loading_state_starts_recording_without_waiting_for_models(self):
        # 起動直後(灰)でも録音は始められる。マイクにモデルは要らない
        fake = _fake(state=app_rt.LOADING)
        MicButton.toggle(fake)
        fake.core.start_recording.assert_called_once()
        fake.state_changed.emit.assert_called_once_with(app_rt.RECORDING)

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


class RecordWhileLoadingTest(unittest.TestCase):
    """モデルロード中に録音を始めたときの取り決め。"""

    def _fake(self, final_ready, models_ready):
        fake = _fake(state=app_rt.BUSY, wake_on=False)
        fake._final_ready = threading.Event()
        fake._models_ready = threading.Event()
        if final_ready:
            fake._final_ready.set()
        if models_ready:
            fake._models_ready.set()
        fake.core.stop_recording.return_value = np.zeros(16000, dtype=np.float32)
        fake.core.transcribe.return_value = "テスト"
        return fake

    def test_finish_waits_for_the_final_model_then_transcribes(self):
        # 停止時に large-v3 が無ければ、ロード完了まで待ってから確定する(捨てない)
        fake = self._fake(final_ready=False, models_ready=False)
        order = []
        fake.core.transcribe.side_effect = lambda a: (order.append("transcribe"), "テスト")[1]

        def loader():
            time.sleep(0.2)
            order.append("ready")
            fake._final_ready.set()

        threading.Thread(target=loader).start()
        MicButton._finish(fake)
        self.assertEqual(order, ["ready", "transcribe"], "モデルが来る前に確定しようとしている")
        fake.core.deliver.assert_called_once_with("テスト")

    def test_finish_tells_the_user_it_is_waiting(self):
        fake = self._fake(final_ready=False, models_ready=False)
        fake._final_ready.set()  # 待ちには入るが即抜ける
        fake._final_ready.clear()
        threading.Timer(0.05, fake._final_ready.set).start()
        MicButton._finish(fake)
        shown = [c.args[0] for c in fake.preview_text.emit.call_args_list]
        self.assertTrue(any("読込中" in s for s in shown), shown)

    def test_finish_does_not_wait_when_the_model_is_already_there(self):
        fake = self._fake(final_ready=True, models_ready=True)
        MicButton._finish(fake)
        shown = [c.args[0] for c in fake.preview_text.emit.call_args_list]
        self.assertFalse(any("読込中" in s for s in shown), shown)
        fake.state_changed.emit.assert_called_with(app_rt.IDLE)

    def test_finish_returns_to_loading_while_models_are_still_arriving(self):
        # large-v3 は来たが small/VAD が未着 → ボタンは灰に戻る(青にしない)
        fake = self._fake(final_ready=True, models_ready=False)
        MicButton._finish(fake)
        fake.state_changed.emit.assert_called_with(app_rt.LOADING)

    def test_loading_state_is_promoted_to_idle_once_models_are_ready(self):
        # _finish の「灰へ戻す」がロード完了通知の後に届いても灰で固まらない
        fake = _fake(wake_on=False)
        fake._models_ready = threading.Event()
        fake._models_ready.set()
        MicButton._on_state(fake, app_rt.LOADING)
        self.assertEqual(fake.state, app_rt.IDLE)

    def test_loading_state_stays_while_models_are_not_ready(self):
        fake = _fake(wake_on=False)
        fake._models_ready = threading.Event()
        MicButton._on_state(fake, app_rt.LOADING)
        self.assertEqual(fake.state, app_rt.LOADING)

    def test_models_ready_does_not_override_an_active_recording(self):
        # ロード完了が録音中に届いても RECORDING を IDLE に戻さない(録音が途切れる)
        fake = _fake(state=app_rt.RECORDING)
        MicButton._on_notify(fake, "__models_ready__")
        fake._on_state.assert_not_called()

    def test_models_ready_turns_the_button_blue_when_idle(self):
        fake = _fake(state=app_rt.LOADING)
        MicButton._on_notify(fake, "__models_ready__")
        fake._on_state.assert_called_once_with(app_rt.IDLE)

    def test_final_model_loads_before_the_preview_model(self):
        # 確定に要る large-v3 を先に。ready フラグは large-v3 直後に立つ
        fake = _fake()
        fake._final_ready = threading.Event()
        fake._models_ready = threading.Event()
        order = []
        fake.core.load_model.side_effect = lambda: order.append("large")
        fake.core.load_preview_model.side_effect = lambda *_: order.append(
            "small" + ("(final ready)" if fake._final_ready.is_set() else ""))
        MicButton._load_models(fake)
        self.assertEqual(order, ["large", "small(final ready)"])
        self.assertTrue(fake._models_ready.is_set())
        fake.notify.emit.assert_called_once_with("__models_ready__")
        fake.state_changed.emit.assert_not_called()

    def test_preview_loop_idles_until_the_preview_model_arrives(self):
        fake = _fake(_preview_running=True)
        fake.core.preview_model = None
        fake.core.snapshot_audio.return_value = np.zeros(16000, dtype=np.float32)
        ticks = []

        def sleep(_):
            ticks.append(1)
            if len(ticks) >= 3:
                fake._preview_running = False

        with mock.patch.object(app_rt.time, "sleep", side_effect=sleep):
            MicButton._preview_loop(fake)
        fake.core.transcribe_preview.assert_not_called()


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
