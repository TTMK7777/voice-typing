# 保守ガイド

このリポ（日本語の音声入力ツール）を、AI を使わずに保守するための案内です。使い方・インストールは [README.md](README.md)、セキュリティの説明は [SECURITY.md](SECURITY.md)、PR の出し方は [CONTRIBUTING.md](CONTRIBUTING.md) が正本です。ここはそれらに載っていない「どこを触れば何が変わるか」をまとめます。

## ファイルの種類

| 種類 | 例 | 何か | 直すときの注意 |
|---|---|---|---|
| 普通のコード | `core.py`・`app_rt.py`・`wake_word.py`・`wake_listener.py` | Python のプログラム。AI なしで動く | 直したら下の「直したあとの確認」を回す |
| テスト | `tests/test_core.py` | 標準ライブラリ `unittest` のテスト | マイク・GPU・Whisper モデルは不要（録音とモデル呼び出しは差し替えてある） |
| 設定・補助 | `requirements.txt`・`vocab.example.txt`・`run_gui.bat`・`create_shortcut.ps1` | 依存パッケージ（バージョン固定）・認識ヒントの見本・起動用の補助 | `requirements.txt` の追加は「軽さを保つ」方針（[CONTRIBUTING.md](CONTRIBUTING.md)）に反するので理由が要る |
| 文書 | `README.md`・`SECURITY.md`・`STATUS.md` | 人が読む説明 | `README.md` の調整表の既定値は手で写してある（下記） |
| AI への指示書 | `AGENTS.md` | AI（Codex 等）がこのリポを扱うときに最初に読む短い指示。レビューコメントを日本語で書く等 | 動作には影響しない。人間が読んでもよい |

テストでもアプリ本体でもない `test_transcribe.py`・`gen_test_wav.ps1`・`test_text.txt` は、作者が認識精度を手で確かめるための補助です（`tests/` の自動テストには含まれない）。実行には NVIDIA GPU とモデルのダウンロードが要るため、この保守ガイドの確認手順には入れていません。

## よくある作業（逆引き）

表の「確かめ方」は、先に「直したあとの確認」の全テストを回してください。「手で合わせる」は、テストが見ていない写し先です。

| やりたいこと | 触るファイル | 手で合わせる場所 | 確かめ方 |
|---|---|---|---|
| 音声コマンドを足す・変える（「クロード、クリア」等） | `core.py` の `VOICE_COMMANDS` と `COMMAND_TRIGGERS`。固定しているテスト: `tests/test_core.py` | `README.md` の「音声コマンド」の表 | 全テスト。コマンド語の名前を変えると `tests/test_core.py` が赤になる |
| 新しい幻覚フレーズ（無音で混入する定型句）を消す | `core.py` の `HALLUCINATIONS` に 1 行足す。固定しているテスト: `tests/test_core.py` | なし | 全テスト。既存の行を消すと `tests/test_core.py` が赤になる |
| ウェイクワードを変える | `wake_word.py` の `_WAKE_RE`（正規化後の文字に対する正規表現）と `WAKE_PROMPT`。固定しているテスト: `tests/test_wake_word.py`、`tests/test_app_wiring.py`（`WAKE_PROMPT` を参照） | `README.md` の「ハンズフリー」の説明文（「やっほークロード」）、`app_rt.py` のボタンのツールチップ（マウスを乗せると出る説明） | 全テスト。`_WAKE_RE` を狭めると `tests/test_wake_word.py` が赤になる。実際に待受が拾うかは実機（マイク）での確認が必要 |
| 発話の拾い始め・切れ目を変える | `wake_listener.py` の `VAD_SPEECH_PROB` / `VAD_SILENCE_PROB`（Silero VAD＝発話区間を見つける軽いモデルの確率しきい値）、`WAKE_JOIN_SEC` | `README.md` の調整表 | 全テストは通るだけ（値そのものは固定していない）。実機で喋って確認が必要 |
| 喋り終わりから貼られるまでの間・息継ぎ区切り・自動終了の時間を変える | `app_rt.py` の `DICTATE_SILENCE_SEC`・`DICTATE_LONG_SPEECH_SEC`・`DICTATE_BREATH_SEC`・`SESSION_IDLE_TIMEOUT_SEC` | `README.md` の調整表と「ハンズフリー」の説明（0.5 秒・30 秒・4 秒・0.25 秒）、`SECURITY.md` の「既知の限界」（30 秒） | 値を変えても全テストは赤にならない（テストは定数を名前で参照しているだけ）。実機で確認が必要 |
| プレビューの速さ・表示文字数・モデルを変える | `app_rt.py` の `PREVIEW_INTERVAL`・`MAX_PREVIEW_CHARS`・`PREVIEW_MODEL` | `README.md` の調整表 | 全テストは赤にならない。実機で確認が必要 |
| ホットキーを変える | `app_rt.py` の `HOTKEY`・`WAKE_HOTKEY`、ボタンだけの簡易版 `app.py` の `HOTKEY` | `README.md`（使い方・ハンズフリー）、`SECURITY.md`（権限の表と既知の限界）、`app_rt.py` と `app.py` のツールチップ・ログ文言（`Ctrl+Alt+Space` と書いてある所） | 全テストは赤にならない。実機で確認が必要 |
| 認識ヒント（よく使う語）を足す | 利用者が `vocab.example.txt` を `vocab.txt` にコピーして編集（`vocab.txt` は個人用で git に載らない） | なし | 1 語 5 トークン前後で 30 語ほどが上限。超えると起動時に `[vocab]` 警告が黒窓に出る |

## 直したあとの確認

1. 依存を入れる（初回のみ）。`requirements.txt` には NVIDIA の CUDA ライブラリ（`nvidia-*`）も入っていて大きい。テストを回すだけなら不要だが、README の手順どおり全部入れるのが基本

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. 全テストを回す。全部 OK（skipped は許容）で合格。マイクや GPU は使わない

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

   依存が入っていないと、`faster_whisper` の import で一部のテストが落ちる（`ModuleNotFoundError`）。コードの不具合ではなく手順 1 の不足。

3. GUI・貼り付け・ホットキーを変えたときは、実機で起動して試す（[CONTRIBUTING.md](CONTRIBUTING.md) の「PR の出し方」4 番）。起動すると画面にマイクボタンが出る。**このガイドの作成時は実機起動をしていない**ため、起動の確認手順は [README.md](README.md) の「起動」に従うこと

CI（自動テスト）は置いていません。理由は [README.md](README.md) の「検討して見送ったこと」にあります。

## 壊れたとき

| 症状 | まず見る所 | よくある原因 |
|---|---|---|
| デスクトップのアイコンから起動すると何も起きない・原因が見えない | `run_gui.bat` から起動し、黒窓のログを見る | アイコン起動（`pythonw.exe`）は黒窓が無く、ログが 1 行も見えない。ファイルへのログ出力は未実装（`STATUS.md` の次の一手） |
| 起動時に CUDA / DLL のエラー | `cuda_setup.py`、`requirements.txt` の `nvidia-*` | `.venv` に CUDA ライブラリ（pip の `nvidia-*`）が入っていない。動作前提は Windows + NVIDIA GPU で、CPU では動かない（`device="cuda"` 固定） |
| テストが `ModuleNotFoundError` で落ちる | 「直したあとの確認」の手順 1 | 依存が入っていない、または `.venv` 以外の Python で回している |
| 認識語がヒントに効かない | 起動時の黒窓の `[vocab]` 警告 | `vocab.txt` が長すぎて先頭が捨てられている。使わない語を消し、大事な語を下に置く |
| ウェイクワードを言っても反応しない・誤発火する | 黒窓の `[wake] '...' -> HIT` / `ignore` の行 | 何と聞き取られたかがそのまま出る。`wake_word.py` の `_WAKE_RE`、`wake_listener.py` の `VAD_SPEECH_PROB` を見る。待受は既定で OFF（`Ctrl+Alt+L`） |
| 貼り付けが遅い・細切れに貼られる | 黒窓の `[timing]` 行 | `DICTATE_SILENCE_SEC`（短いと文中の間で切れ、長いと遅い）。内訳は音声長・確定・貼付 |
| アンチウイルスが検知する | [SECURITY.md](SECURITY.md) の「アンチウイルス／EDR が誤検知する可能性について」 | キー入力監視＋キーストローク送出＋クリップボード操作の組み合わせが誤検知されやすい |

## AI なしでは保守できない所

なし。このリポは普通の Python プログラムで、AI 専用の部品（スキル・hook 等）はありません（`AGENTS.md` は AI 用の短いメモで、無くても動作に影響しない）。

ただし次は人が実機で確かめるしかありません。自動テストでは担保できません。

- 実際の声での認識精度（ウェイクワードの取りこぼし・誤発火、息継ぎ区切りで文が途中で切れないか）。`STATUS.md` の判断待ちにも同じ項目がある
- GUI の見た目、ホットキー、他アプリへの貼り付け

## 文書間の食い違い

- `README.md` と `CONTRIBUTING.md` は「現状で完成。機能追加は基本しない」と書き、`STATUS.md` は「機能追加が続いている個人ツール」と書いている。コードの履歴ではウェイクワード改善などの機能変更が最近も入っているため、実態は「基本は追加しないが、作者の必要に応じて入る」。迷ったら `README.md` の「このプロジェクトの位置づけ」を現行の方針として扱う
- `SECURITY.md` §3 に書かれた各ファイルの行数は、現在のファイルと一致しない（行数は変わるため）。数字は目安として読むこと

## 用語集

| 言葉 | 意味 |
|---|---|
| Whisper / faster-whisper | 音声を文字にする AI モデルと、それを速く動かすライブラリ。本ツールは `large-v3`（確定用）と `small`（暫定表示用）を使う |
| CUDA | NVIDIA GPU で計算するための仕組み。本ツールは GPU 前提 |
| VAD（Silero VAD） | 「いま人が喋っているか」を判定する軽いモデル。発話区間の切り出しに使う |
| ウェイクワード | 待受を起動する合図の言葉（「やっほークロード」）。「クロード」は本ツールの呼び名で、特定の製品を呼ぶものではない |
| 幻覚（ハルシネーション） | 無音や語尾で Whisper が勝手に出す定型句（「ご視聴ありがとうございました」等）。`HALLUCINATIONS` で取り除く |
| initial_prompt（認識ヒント） | 文字起こしの前に渡す語の一覧。`vocab.txt` の中身がここに入る |
| トークン | モデルが文字を数える単位。日本語の固有名詞で 1 語あたり約 5 |
| 待受 / 連続口述 | 待受＝ウェイクワードを待つ状態（ボタン緑）。連続口述＝喋るたびに貼り付ける状態（ボタン赤） |
| EDR | パソコンの不審な動きを監視するセキュリティ製品。本ツールの動作が誤検知されることがある |
| スキル・hook・MCP・frontmatter | このリポには存在しません（AI 開発環境の用語で、本ツールの保守には不要） |
