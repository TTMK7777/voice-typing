"""ウェイクワード「やっほー Claude」の判定(純ロジック)。

このモジュールは OS・マイク・GPU・モデルに一切触れない。文字列を受け取って
真偽を返すだけなので、GPU の無い環境でも単体テストできる。

Whisper は同じ発話でも表記が揺れる(「ヤッホー、クロード」「やっほークラウド」
「Yahoo Claude」等)。そこで正規化してから緩めのパターンで照合する:

  1. NFKC 正規化(全角英数 → 半角)+ 小文字化
  2. カタカナ → ひらがな
  3. 長音「ー」・促音「っ」・記号・空白を除去

これで「ヤッホー」「やっほー」「やほー」はすべて "やほ" に潰れる。
"""
import re
import unicodedata

_KATAKANA_START, _KATAKANA_END = 0x30A1, 0x30F6
_KANA_OFFSET = 0x60

# 発話の揺れとして落とす文字(長音・促音・区切り記号・空白)
_DROPPED = set("っー〜~・､、。,.!！?？:：;；「」『』()（） 　\t\r\n\"'")

# 正規化後の照合パターン。
# 先頭 3 文字までの前置き(「あ、やっほークロード」等)は許容する。
#
# 前半「やっほー」: Whisper は「やは」とも書き起こす(実機ログ: 'やはー、グロード'
#   'やはくロード')ため や[はほ] まで許容する。ただし「やはり」は日常語で、
#   「やはりクロード」を拾うと誤発火するので (?!り) で明示的に外す。
# 後半「クロード」: 「クラウド」「クラドウ」「グロード」等に化ける。
_WAKE_RE = re.compile(
    r"^.{0,3}(や[はほ](?!り)|yaho+)う?.{0,4}(くろ|ぐろ|くら[うど]|claude|cloud)"
)


def normalize(text):
    """表記揺れを吸収した比較用の文字列にする。"""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    chars = []
    for ch in normalized:
        if _KATAKANA_START <= ord(ch) <= _KATAKANA_END:
            ch = chr(ord(ch) - _KANA_OFFSET)  # カタカナ → ひらがな
        if ch in _DROPPED:
            continue
        chars.append(ch)
    return "".join(chars)


def is_wake(text):
    """認識テキストがウェイクワードで始まっていれば True。

    例: 「ヤッホー、クロード」→ True
        「やっほークラウド。」→ True
        「クロード、クリア」  → False (ウェイクワードではなく音声コマンド)
        「やっほー、元気?」   → False
    """
    return bool(_WAKE_RE.match(normalize(text)))
