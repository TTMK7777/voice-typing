"""音声入力のコアロジック(録音・文字起こし・貼り付け)。CLI版/GUI版で共有。"""
import threading
import time

import cuda_setup  # noqa: F401  CUDA DLL パス登録(faster_whisper より前)
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard

SAMPLE_RATE = 16000

# 貼り付け後、クリップボードを元の内容へ戻すまでの待ち時間(秒)。
# 対象アプリが Ctrl+V を処理し終える前に復元してしまうと、復元後の「古い内容」が
# 貼られてしまう。Windows には「クリップボードが読まれた」を知る一般的な手段が
# 無いため、ここは確率的な緩和にとどまる(残存リスクは SECURITY.md「既知の限界」)。
# 遅いアプリ(Electron 系・リモートデスクトップ)でも間に合うよう余裕を持たせている。
PASTE_SETTLE_SEC = 0.4

# ウェイクワード待受でマイクを監視する際の1ブロックの長さ(秒)。
# 短いほど反応が速いがコールバック回数が増える。
MONITOR_BLOCK_SEC = 0.1

# ===== 音声コマンド =====
# 「<トリガー語>、<コマンド語>」の発話を検知したら、テキストを貼り付ける代わりに
# スラッシュコマンドを注入して Enter で実行する(Claude Code の /clear /compact 等を
# キーボードなしで実行するための層)。
# 誤発火対策: トリガー語必須 + コマンド語の後に余計な語が続く場合は
# 引数対応コマンド(コンパクト)以外はコマンド扱いしない。
COMMAND_TRIGGERS = ("クロード", "くろーど", "claude")  # Whisper の表記揺れを吸収(小文字比較)
# コマンド語 → (引数なし時のコマンド, 引数あり時のテンプレート or None)
VOICE_COMMANDS = {
    "クリア": ("/clear", None),
    "コンパクト": ("/compact", "/compact Focus on {arg}"),
}
_CMD_SEPARATORS = "、。，．,.!！?？・:：;； 　\t"


def match_voice_command(text):
    """認識テキストが音声コマンドなら注入するコマンド文字列を返す。非該当は None。

    例: 「クロード、クリア。」→ "/clear"
        「クロード、コンパクト、n8nの話」→ "/compact Focus on n8nの話"
        「クロードに聞いてみよう」→ None (通常の口述として貼り付け)
    """
    t = (text or "").strip().strip(_CMD_SEPARATORS)
    lowered = t.lower()
    rest = None
    for trig in COMMAND_TRIGGERS:
        if lowered.startswith(trig):
            rest = t[len(trig):].lstrip(_CMD_SEPARATORS)
            break
    if rest is None:
        return None
    for word, (cmd, arg_template) in VOICE_COMMANDS.items():
        if rest == word:
            return cmd
        if rest.startswith(word):
            arg = rest[len(word):].strip(_CMD_SEPARATORS)
            if arg and arg_template:
                return arg_template.format(arg=arg)
            return None  # 「クリアしてください」等は誤認識の可能性があるため実行しない
    return None


# initial_prompt に入れられるトークン数の上限。
# Whisper は prompt の**末尾** max_length//2 - 1 トークンだけを使い、あふれた先頭は
# 警告なく捨てる(faster_whisper/transcribe.py の get_prompt)。捨てられても
# エラーは出ないので、vocab.txt に語を足し続けると「足したのに効かない」「先に
# 句読点誘導文だけ消える」が無言で起きる。ここで枠内に収め、落とした語は必ず知らせる。
#
# 実測(large-v3 / 2026-08-13): 句読点誘導文だけで 38 トークン、日本語の固有名詞は
# 1 語あたり約 5 トークン。つまり語彙に使えるのは 30 語ほどが上限。
PROMPT_TOKEN_LIMIT = 223

# Whisper 日本語の定番幻覚(無音/末尾で混入する学習データ=YouTube字幕由来の定型句)
HALLUCINATIONS = [
    "最後までご視聴いただきありがとうございました",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご清聴ありがとうございました",
    "ご清聴ありがとうございます",
    "高評価とチャンネル登録をお願いします",
    "チャンネル登録をお願いします",
    "チャンネル登録よろしくお願いします",
    "次回の動画でお会いしましょう",
]


class VoiceCore:
    def __init__(self, model_size="large-v3", language="ja",
                 vocab_file="vocab.txt", restore_clipboard=True):
        self.model_size = model_size
        self.language = language
        self.vocab_file = vocab_file
        self.restore_clipboard = restore_clipboard
        self.model = None
        self.preview_model = None
        self.recording = False
        self._frames = []
        self._stream = None
        self._monitor_stream = None
        self._monitor_cb = None
        self._restore_thread = None
        self._kb = keyboard.Controller()
        self.initial_prompt = self._build_prompt()

    def _load_vocab(self):
        """vocab.txt があれば認識ヒント(initial_prompt)として読み込む。"""
        words = self._load_vocab_words()
        return "、".join(words) if words else None

    def _load_vocab_words(self):
        """vocab.txt の語を1行1語で読む(空行と # 始まりは無視)。無ければ空リスト。"""
        try:
            with open(self.vocab_file, encoding="utf-8") as f:
                return [s for s in (line.strip() for line in f)
                        if s and not s.startswith("#")]
        except FileNotFoundError:
            return []

    def reload_vocab(self):
        self.initial_prompt = self._build_prompt()

    def count_tokens(self, text):
        """text のトークン数。モデル未ロード時は文字数で概算する。

        日本語では文字数がトークン数をわずかに上回る(実測 114 文字 = 108 トークン)
        ため、概算は安全側(多め)に振れる。
        """
        if self.model is not None:
            try:
                return len(self.model.hf_tokenizer.encode(
                    " " + text.strip(), add_special_tokens=False).ids)
            except Exception:
                pass  # トークナイザが取れない版でも概算で続行する
        return len(text)

    def _fit_vocab(self, base, words):
        """語彙を PROMPT_TOKEN_LIMIT の残り枠に収める。(採用した語, 落とした語) を返す。

        枠を超えた分は Whisper 側で**先頭から**捨てられるので、こちらも先頭から
        落として挙動を一致させる。結果として vocab.txt の後ろに書いた語ほど残る。
        句読点誘導文(base)は機能なので必ず確保し、削るのは語彙側だけにする。
        """
        head = base + "登場する固有名詞: "
        budget = PROMPT_TOKEN_LIMIT - self.count_tokens(head + "。")
        kept = []
        used = 0
        for word in reversed(words):
            cost = self.count_tokens(word + "、")
            if used + cost > budget:
                break
            kept.insert(0, word)
            used += cost
        return kept, words[:len(words) - len(kept)]

    def _build_prompt(self):
        """句読点を誘導する自然文 + 語彙ヒントを initial_prompt にする。
        Whisper は直前文脈の文体を真似るので、句読点付きの文を渡すと句読点が出やすい。"""
        base = "以下は日本語の音声入力です。句読点を適切に付けて、自然な文章に書き起こします。"
        words = self._load_vocab_words()
        if not words:
            return base
        kept, dropped = self._fit_vocab(base, words)
        if dropped:
            # 黙って効かなくなるのが最悪なので、必ず知らせる。全部並べると読めないので
            # 落ちた語の先頭だけ具体名を出す(どこから切れたかが分かれば直せる)。
            head = "、".join(dropped[:5])
            more = f" ほか {len(dropped) - 5} 語" if len(dropped) > 5 else ""
            print(f"[vocab] {self.vocab_file} が長すぎます。先頭の {len(dropped)} 語は"
                  f"認識ヒントに入りません: {head}{more}")
            print(f"[vocab] 効くのは末尾の {len(kept)} 語だけです"
                  f"(上限 {PROMPT_TOKEN_LIMIT} トークン)。よく使う語を後ろに置いてください。")
        if not kept:
            return base
        return base + "登場する固有名詞: " + "、".join(kept) + "。"

    def load_model(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        # __init__ の時点ではトークナイザが無く文字数の概算だったので、
        # 実トークン数で組み直す(概算は安全側=多めなので、ここで枠が広がる)。
        self.reload_vocab()

    def warmup(self):
        """ロード済みモデルに無音を1回通し、初回推論の初期化コスト(実測 0.5s)を
        起動時に前払いする。ユーザーの最初の口述がその分だけ遅れないようにするため。"""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if self.model is not None:
            list(self.model.transcribe(silence, language=self.language, beam_size=1)[0])
        if self.preview_model is not None:
            list(self.preview_model.transcribe(silence, language=self.language, beam_size=1)[0])

    def _callback(self, indata, frame_count, time_info, status):
        if self.recording:
            self._frames.append(indata.copy())

    def start_recording(self):
        self._frames = []
        self.recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop_recording(self):
        """録音停止し音声(np.ndarray)を返す。無音なら None。"""
        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return None
        return np.concatenate(self._frames, axis=0).flatten()

    # ---- ウェイクワード待受用のマイク監視 ----
    # 録音(start_recording)とは別のストリーム。両方を同時に開くことはなく、
    # 録音を始める前に必ず stop_monitor() で閉じる(GUI 側が保証)。
    # 受け取った音声はメモリ上のバッファのみで、保存も送信もしない。
    def start_monitor(self, on_block):
        """マイクの常時監視を開始し、音声ブロックを on_block(np.ndarray) に渡す。

        すでに監視中なら**ストリームは開いたまま渡し先だけ差し替える**。
        ウェイクワード待受 → 連続口述の切り替えでマイクを閉じて開き直すと、
        その間の音が落ちて発話の頭が欠けるため。
        """
        self._monitor_cb = on_block
        if self._monitor_stream is not None:
            return
        self._monitor_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=int(SAMPLE_RATE * MONITOR_BLOCK_SEC),
            callback=self._monitor_callback,
        )
        self._monitor_stream.start()

    def _monitor_callback(self, indata, frame_count, time_info, status):
        cb = self._monitor_cb
        if cb is None:
            return
        try:
            cb(indata.copy().flatten())
        except Exception:
            pass  # オーディオコールバックで例外を投げるとストリームごと死ぬため握り潰す

    def stop_monitor(self):
        """マイクの常時監視を停止する(停止済みなら何もしない)。"""
        self._monitor_cb = None
        if self._monitor_stream is not None:
            self._monitor_stream.stop()
            self._monitor_stream.close()
            self._monitor_stream = None

    def _strip_hallucinations(self, text):
        """無音/末尾に出る Whisper の定番幻覚フレーズを除去する。"""
        for h in HALLUCINATIONS:
            text = text.replace(h, "")
        return text.strip()

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language=self.language, initial_prompt=self.initial_prompt,
            vad_filter=True, beam_size=5,
        )
        return self._strip_hallucinations(
            "".join(seg.text for seg in segments).strip()
        )

    @staticmethod
    def _safe_copy(text):
        """pyperclip.copy を例外で握りつぶす(Windows の一時的な clipboard ロックで
        PyperclipWindowsException が飛んでもアプリを落とさない)。成否を返す。"""
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    def paste(self, text):
        """クリップボード経由でアクティブウィンドウのカーソル位置に貼り付け。"""
        old = None
        if self.restore_clipboard:
            try:
                old = pyperclip.paste()
            except Exception:
                old = None
        if not self._safe_copy(text):
            return  # copy 失敗時は Ctrl+V を送らない(古いクリップ内容の誤貼り付けを防ぐ)
        time.sleep(0.05)
        self._kb.press(keyboard.Key.ctrl)
        self._kb.press("v")
        self._kb.release("v")
        self._kb.release(keyboard.Key.ctrl)
        # ここで貼り付けは完了している。復元待ちは別スレッドへ回す:
        # 連続口述では発話を1件ずつ直列に処理するため、ここで PASTE_SETTLE_SEC を
        # 待つと次の発話の文字起こしがその分だけ後ろにずれる。
        if self.restore_clipboard and old is not None:
            self._restore_thread = threading.Thread(
                target=self._restore_after_settle, args=(old, text), daemon=True)
            self._restore_thread.start()

    def _restore_after_settle(self, old, pasted):
        """貼り付け先が Ctrl+V を処理し終える頃合いを待ってから復元する。"""
        time.sleep(PASTE_SETTLE_SEC)
        self._restore_clipboard(old, pasted)

    def _restore_clipboard(self, old, pasted):
        """貼り付け後にクリップボードを元の内容へ戻す。

        自分が書いた内容がまだ残っているときだけ戻す。復元を待つ間に別プロセス
        (ユーザー自身の Ctrl+C、クリップボード管理ツール等)が新しい内容を置いた
        場合、無条件に復元するとその新しい内容を古い内容で踏み潰してしまうため。
        """
        try:
            current = pyperclip.paste()
        except Exception:
            return  # 読めないなら触らない(壊すより何もしない方が安全)
        if current != pasted:
            return  # 他が書き換えている → 復元しない
        self._safe_copy(old)

    def _press_enter(self):
        self._kb.press(keyboard.Key.enter)
        self._kb.release(keyboard.Key.enter)

    def run_voice_command(self, command):
        """スラッシュコマンドを貼り付け、Enter を送って実行する。

        Claude Code の入力欄は "/" で補完ポップアップが開くことがあるため
        Enter を2回(間隔をあけて)送る: 1回目が補完選択に食われても2回目で
        実行され、1回目で実行済みなら空プロンプトへの2回目は無害。
        """
        self.paste(command)
        time.sleep(0.3)
        self._press_enter()
        time.sleep(0.2)
        self._press_enter()

    def deliver(self, text):
        """認識テキストを届ける: 音声コマンドなら実行、通常文なら貼り付け。"""
        command = match_voice_command(text)
        if command:
            print(f"[voice-command] {text!r} → {command}")
            self.run_voice_command(command)
        else:
            self.paste(text)

    # ---- リアルタイム暫定変換(録音中に逐次呼ぶ) ----
    def load_preview_model(self, size="small"):
        from faster_whisper import WhisperModel
        self.preview_model = WhisperModel(size, device="cuda", compute_type="float16")

    def snapshot_audio(self):
        """録音中の現時点までの音声を返す(録音は止めない)。"""
        if not self._frames:
            return None
        return np.concatenate(self._frames, axis=0).flatten()

    def transcribe_preview(self, audio):
        """暫定表示用の高速変換(精度より速度優先)。"""
        # プレビューは速報優先。プロンプトを入れると幻覚/空を招くため付けない
        segments, _ = self.preview_model.transcribe(
            audio, language=self.language, beam_size=1,
            condition_on_previous_text=False,
        )
        return self._strip_hallucinations(
            "".join(seg.text for seg in segments).strip()
        )
