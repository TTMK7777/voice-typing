"""core ロジックのテスト(録音・文字起こし・貼り付け)。

追加依存ゼロ(stdlib unittest + unittest.mock)。実行:
    .venv/Scripts/python.exe -m unittest discover -s tests -v

検証の要点:
- paste() の clipboard 書込が失敗してもクラッシュしない / Ctrl+V を送らない。
- _strip_hallucinations が Whisper の定番幻覚フレーズを除去する。

GPU / マイク / クリップボードは不要。pyperclip と pynput は全てモックする
(本物の Ctrl+V を端末へ送らないため _kb も必ずモックする)。
"""
import sys
import os
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
from core import VoiceCore  # noqa: E402


class PasteSafetyTest(unittest.TestCase):
    def test_safe_copy_swallows_exception(self):
        with mock.patch.object(core.pyperclip, "copy", side_effect=RuntimeError("clipboard locked")):
            self.assertFalse(VoiceCore._safe_copy("x"))

    def test_safe_copy_returns_true_on_success(self):
        with mock.patch.object(core.pyperclip, "copy") as cp:
            self.assertTrue(VoiceCore._safe_copy("x"))
            cp.assert_called_once_with("x")

    def test_paste_skips_ctrl_v_when_clipboard_fails(self):
        """copy 失敗時はクラッシュせず、かつ Ctrl+V を送らない(古い内容の誤貼り付け防止)。"""
        vc = VoiceCore()
        vc._kb = mock.MagicMock()                 # 本物の Ctrl+V を端末へ送らない
        with mock.patch.object(core.time, "sleep"), \
             mock.patch.object(core.pyperclip, "paste", return_value=""), \
             mock.patch.object(core.pyperclip, "copy", side_effect=RuntimeError("locked")):
            vc.paste("貼り付けるテキスト")        # 例外が伝播しなければ成功
        vc._kb.press.assert_not_called()          # copy 失敗時は Ctrl+V を送らない

    def test_paste_sends_ctrl_v_on_success(self):
        """copy 成功時は Ctrl+V を送る(正常系の回帰検出)。"""
        vc = VoiceCore()
        vc._kb = mock.MagicMock()
        with mock.patch.object(core.time, "sleep"), \
             mock.patch.object(core.pyperclip, "paste", return_value="old"), \
             mock.patch.object(core.pyperclip, "copy"):
            vc.paste("貼り付けるテキスト")
        self.assertTrue(vc._kb.press.called)      # Ctrl+V 送出

    def test_paste_restores_previous_clipboard(self):
        """通常時は貼り付け後に元のクリップボード内容へ戻す。"""
        vc = VoiceCore()
        vc._kb = mock.MagicMock()
        # paste(): ①復元用に元内容を読む ②復元直前に現在の内容を読む
        with mock.patch.object(core.time, "sleep"), \
             mock.patch.object(core.pyperclip, "paste",
                               side_effect=["元の内容", "貼り付けるテキスト"]), \
             mock.patch.object(core.pyperclip, "copy") as cp:
            vc.paste("貼り付けるテキスト")
            vc._restore_thread.join(timeout=2)  # 復元は別スレッド(貼り付けを待たせないため)
        self.assertEqual(cp.call_args_list[-1], mock.call("元の内容"))

    def test_paste_does_not_block_on_clipboard_restore(self):
        """復元待ちで貼り付けを足止めしない。

        連続口述では発話を1件ずつ直列に処理するため、ここで PASTE_SETTLE_SEC を
        待つと次の発話の文字起こしがその分だけ後ろにずれる。待ちは別スレッドで行う。
        """
        vc = VoiceCore()
        vc._kb = mock.MagicMock()
        slept = []
        main = threading.current_thread().name

        def fake_sleep(seconds):
            slept.append((threading.current_thread().name, seconds))

        with mock.patch.object(core.time, "sleep", side_effect=fake_sleep),              mock.patch.object(core.pyperclip, "paste",
                               side_effect=["元の内容", "貼り付けるテキスト"]),              mock.patch.object(core.pyperclip, "copy"):
            vc.paste("貼り付けるテキスト")
            vc._restore_thread.join(timeout=2)

        on_main = [s for t, s in slept if t == main and s == core.PASTE_SETTLE_SEC]
        anywhere = [s for _, s in slept if s == core.PASTE_SETTLE_SEC]
        self.assertFalse(on_main, "復元待ちが貼り付けを足止めしている")
        self.assertTrue(anywhere, "復元前の待ちが消えている(早すぎる復元で誤貼り付けの恐れ)")

    def test_paste_does_not_clobber_clipboard_changed_by_others(self):
        """復元直前に別プロセスが新しい内容を置いていたら、復元して壊さない。

        ユーザー自身の Ctrl+C やクリップボード管理ツールが割り込む場合がある。
        ここで無条件に復元すると、その新しい内容を古い内容で踏み潰してしまう。
        """
        vc = VoiceCore()
        vc._kb = mock.MagicMock()
        with mock.patch.object(core.time, "sleep"), \
             mock.patch.object(core.pyperclip, "paste",
                               side_effect=["元の内容", "別プロセスが置いた新しい内容"]), \
             mock.patch.object(core.pyperclip, "copy") as cp:
            vc.paste("貼り付けるテキスト")
            vc._restore_thread.join(timeout=2)
        copied = [c.args[0] for c in cp.call_args_list]
        self.assertNotIn("元の内容", copied)      # 復元しない
        self.assertEqual(copied, ["貼り付けるテキスト"])


class HallucinationStripTest(unittest.TestCase):
    def test_strips_known_hallucination(self):
        vc = VoiceCore()
        self.assertEqual(
            vc._strip_hallucinations("今日の議題です。ご視聴ありがとうございました"),
            "今日の議題です。",
        )

    def test_keeps_normal_text(self):
        vc = VoiceCore()
        self.assertEqual(vc._strip_hallucinations("普通の文章です。"), "普通の文章です。")


class VoiceCommandMatchTest(unittest.TestCase):
    """音声コマンド照合(「クロード、クリア」→ /clear)。"""

    def test_clear_with_punctuation(self):
        self.assertEqual(core.match_voice_command("クロード、クリア。"), "/clear")

    def test_clear_without_punctuation(self):
        self.assertEqual(core.match_voice_command("クロードクリア"), "/clear")

    def test_compact_plain(self):
        self.assertEqual(core.match_voice_command("クロード、コンパクト"), "/compact")

    def test_compact_with_topic_arg(self):
        self.assertEqual(
            core.match_voice_command("クロード、コンパクト、n8nの話"),
            "/compact Focus on n8nの話",
        )

    def test_ascii_trigger_variant(self):
        self.assertEqual(core.match_voice_command("Claude、クリア"), "/clear")

    def test_normal_sentence_starting_with_trigger_is_not_command(self):
        self.assertIsNone(core.match_voice_command("クロードに聞いてみようと思います。"))

    def test_clear_with_trailing_words_rejected(self):
        """引数非対応コマンドに続きが付く場合は誤認識の可能性 → 実行しない。"""
        self.assertIsNone(core.match_voice_command("クロード、クリアしてください"))

    def test_command_word_without_trigger_rejected(self):
        self.assertIsNone(core.match_voice_command("クリア"))

    def test_empty_and_none(self):
        self.assertIsNone(core.match_voice_command(""))
        self.assertIsNone(core.match_voice_command(None))


class VoiceCommandRunTest(unittest.TestCase):
    """コマンド実行経路(deliver / run_voice_command)。_kb は必ずモックする。"""

    def _core(self):
        vc = VoiceCore()
        vc._kb = mock.MagicMock()
        return vc

    def test_deliver_runs_command_instead_of_paste(self):
        vc = self._core()
        with mock.patch.object(vc, "run_voice_command") as run, \
             mock.patch.object(vc, "paste") as paste:
            vc.deliver("クロード、クリア")
        run.assert_called_once_with("/clear")
        paste.assert_not_called()

    def test_deliver_pastes_normal_text(self):
        vc = self._core()
        with mock.patch.object(vc, "run_voice_command") as run, \
             mock.patch.object(vc, "paste") as paste:
            vc.deliver("普通の口述テキストです。")
        run.assert_not_called()
        paste.assert_called_once_with("普通の口述テキストです。")

    def test_run_voice_command_pastes_then_sends_enter_twice(self):
        vc = self._core()
        with mock.patch.object(core.time, "sleep"), \
             mock.patch.object(vc, "paste") as paste:
            vc.run_voice_command("/clear")
        paste.assert_called_once_with("/clear")
        enters = [c for c in vc._kb.press.call_args_list
                  if c.args[0] == core.keyboard.Key.enter]
        self.assertEqual(len(enters), 2)


class VocabPromptBudgetTest(unittest.TestCase):
    """語彙ヒントが Whisper の prompt 枠(末尾 223 トークン)から溢れないこと。

    溢れた分は Whisper 側で警告なく捨てられ、「vocab.txt に足したのに効かない」
    が無言で起きる。ここでは実測に合わせトークン数=文字数の概算で検証する
    (モデル未ロード時の経路と同じ)。
    """

    def _core(self, words, tmp):
        path = os.path.join(tmp, "vocab.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(words))
        with mock.patch.object(core.keyboard, "Controller"):
            return VoiceCore(vocab_file=path)

    def test_short_vocab_is_kept_whole(self):
        import tempfile
        words = ["山王病院", "つじラボ", "オリコン"]
        with tempfile.TemporaryDirectory() as tmp:
            vc = self._core(words, tmp)
        for w in words:
            self.assertIn(w, vc.initial_prompt)

    def test_prompt_stays_within_token_limit(self):
        import tempfile
        words = [f"検証用語彙{i:03d}" for i in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            vc = self._core(words, tmp)
        self.assertLessEqual(vc.count_tokens(vc.initial_prompt),
                             core.PROMPT_TOKEN_LIMIT)

    def test_punctuation_hint_survives_a_huge_vocab(self):
        """語彙が溢れても句読点誘導文は残る(これが消えると句読点が付かなくなる)。"""
        import tempfile
        words = [f"検証用語彙{i:03d}" for i in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            vc = self._core(words, tmp)
        self.assertIn("句読点を適切に付けて", vc.initial_prompt)

    def test_last_words_win_when_vocab_overflows(self):
        """溢れたときに残るのは末尾の語(Whisper が末尾から採るのに合わせる)。"""
        import tempfile
        words = [f"検証用語彙{i:03d}" for i in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            vc = self._core(words, tmp)
        self.assertIn("検証用語彙199", vc.initial_prompt)
        self.assertNotIn("検証用語彙000", vc.initial_prompt)

    def test_overflow_is_reported(self):
        """黙って捨てない: 落ちた語を必ず知らせる。"""
        import tempfile
        words = [f"検証用語彙{i:03d}" for i in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print") as p:
                self._core(words, tmp)
        said = " ".join(str(c) for c in p.call_args_list)
        self.assertIn("検証用語彙000", said)

    def test_no_report_when_vocab_fits(self):
        """収まっているのに警告を出さない(狼少年にしない)。"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print") as p:
                self._core(["山王病院", "つじラボ"], tmp)
        p.assert_not_called()


if __name__ == "__main__":
    unittest.main()
