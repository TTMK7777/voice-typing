"""
リアルタイム版: フローティングのマイクボタン + 暫定テキストのプレビュー。

- 録音中、small モデルで PREVIEW_INTERVAL 秒ごとに暫定テキストを更新表示(揺れる)
- 停止で large-v3 が高精度確定 → アクティブウィンドウに自動入力
- 起動直後(灰=モデルロード中)でも録音は始められる。マイクにモデルは要らないので、
  ロードは録音の裏で進み、停止時にまだ無ければそこで待つ(待つのは差分だけ)
- ボタン/プレビューともフォーカスを奪わない(入力先がズレない)
"""
import sys
import time
import threading
import ctypes
from ctypes import wintypes

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QFont, QPen, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QSystemTrayIcon, QMenu
from pynput import keyboard as pynput_keyboard

from core import VoiceCore, SAMPLE_RATE
from wake_listener import WakeDetector, SpeechSegmenter, safe_log

IDLE, LOADING, RECORDING, BUSY = "idle", "loading", "recording", "busy"
SESSION = "session"                    # 連続口述セッション中(喋る→黙る→貼る の繰り返し)
LISTENING = "listening"                # ウェイクワード待受中(state ではなく IDLE の表示色)
COLORS = {
    IDLE: QColor("#2d7dd2"),
    LOADING: QColor("#868e96"),
    RECORDING: QColor("#e03131"),
    BUSY: QColor("#f08c00"),
    SESSION: QColor("#e03131"),
    LISTENING: QColor("#2f9e44"),
}
HOTKEY = "<ctrl>+<alt>+<space>"        # 生口述(録音 開始/停止)
WAKE_HOTKEY = "<ctrl>+<alt>+l"         # ウェイクワード待受の ON/OFF
BTN_SIZE = 64

# ===== 録音中の Enter キーで停止 =====
# フォーカス中の入力欄(チャット欄等)への誤送信/誤改行を防ぐため、
# 低レベルフックで「録音中に限り」Enter キーが他アプリへ伝播するのを遮断する。
VK_RETURN = 0x0D
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_KEYUP = 0x0101
WM_SYSKEYUP = 0x0105

user32 = ctypes.windll.user32


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


_HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.restype = ctypes.c_void_p
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, _HOOKPROC, ctypes.c_void_p, wintypes.DWORD]
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]

# ===== チューニング用パラメータ(ここをいじって最適解を探す) =====
PREVIEW_INTERVAL = 0.2     # 暫定変換の最短間隔(秒)。変換が速ければこの間隔で更新
MAX_PREVIEW_CHARS = 120    # プレビューに表示する末尾文字数(あふれ防止)
PREVIEW_MODEL = "small"    # 暫定モデル。"tiny" にすると更に高速(精度は落ちる/確定で直る)

# ===== 連続口述セッション(ウェイクワードで始まり、喋る→黙る→貼る を繰り返す) =====
# ここの無音の長さがそのまま「喋り終わってから貼られるまで」の体感になる。
# 短いほど自然だが、文の途中の「間」で区切られて細切れに貼られやすくなる。
DICTATE_SILENCE_SEC = 0.5      # これだけ黙ったら、そこまでを1つの発話として貼り付ける
MIN_UTTERANCE_SEC = 0.5        # これより短い音は貼らない(相槌・物音で貼られないように)
MAX_UTTERANCE_SEC = 60.0       # 1発話の上限(区切らず喋り続けた場合の保険)
SESSION_IDLE_TIMEOUT_SEC = 30.0  # 何も喋らないままこの時間たったらセッションを終える
UTTERANCE_QUEUE = 8            # 文字起こし待ちの発話をためておける数


def apply_noactivate(widget):
    """フォーカスを奪わないウィンドウにする(Windows)。show 後に呼ぶ。"""
    try:
        GWL_EXSTYLE = -20
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32
        ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
    except Exception:
        pass


class PreviewWindow(QWidget):
    """録音中だけ出る暫定テキストの吹き出し。"""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.label = QLabel("…", self)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setStyleSheet(
            "background: rgba(20,20,20,225); color: white;"
            "border-radius: 12px; padding: 12px; font-size: 15px;"
        )
        self.resize(380, 130)
        self._drag = None
        self.user_moved = False

    def set_text(self, text):
        if text and len(text) > MAX_PREVIEW_CHARS:
            text = "…" + text[-MAX_PREVIEW_CHARS:]
        self.label.setText(text if text else "…")
        self.label.resize(self.size())

    def place_above(self, btn_geo):
        screen = QApplication.primaryScreen().availableGeometry()
        x = btn_geo.center().x() - self.width() // 2
        y = btn_geo.top() - self.height() - 10
        # 画面内に収める(右端・左端・上端・下端をクランプ)
        x = max(screen.left() + 10, min(x, screen.right() - self.width() - 10))
        if y < screen.top() + 10:
            y = btn_geo.bottom() + 10  # 上に入らなければボタンの下に出す
        y = min(y, screen.bottom() - self.height() - 10)
        self.move(x, y)

    # ---- ドラッグ移動(一度動かすと以降その位置を維持) ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag is not None and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag)
            self.user_moved = True
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag = None
        e.accept()

    def showEvent(self, e):
        super().showEvent(e)
        self.label.resize(self.size())
        apply_noactivate(self)


class MicButton(QWidget):
    state_changed = Signal(str)
    preview_text = Signal(str)
    notify = Signal(str)

    def __init__(self):
        super().__init__()
        self.state = LOADING
        self.core = VoiceCore()
        self.preview = PreviewWindow()
        self._drag_pos = None
        self._moved = False
        self._preview_running = False
        # ウェイクワード待受: small モデルを暫定プレビューと共用するため排他する
        self._model_lock = threading.Lock()
        # 確定モデル(large-v3)が使えるようになった / 全モデル+VAD が揃った
        self._final_ready = threading.Event()
        self._models_ready = threading.Event()
        self.wake_on = False
        self._wake = WakeDetector(
            transcribe=self._wake_transcribe,
            on_wake=lambda: self.notify.emit("__wake__"),
            sample_rate=SAMPLE_RATE,
            log=safe_log,
        )
        # 連続口述: 喋り終わりの無音で区切り、区間ごとに確定 → 貼り付け
        self._dictate = SpeechSegmenter(
            self._on_utterance,
            sample_rate=SAMPLE_RATE,
            silence_hold_sec=DICTATE_SILENCE_SEC,
            min_speech_sec=MIN_UTTERANCE_SEC,
            max_segment_sec=MAX_UTTERANCE_SEC,
            check_interval_sec=None,   # 途中で区切らない(黙るまでが1発話)
            queue_size=UTTERANCE_QUEUE,
            log=safe_log,
        )

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setToolTip(
            "クリック / Ctrl+Alt+Space = 録音 開始/停止\n"
            "Ctrl+Alt+L = ウェイクワード待受(「やっほークロード」)の ON/OFF\n"
            "緑 = 待受中 / 赤 = 連続口述中(喋ると貼られます)\n"
            "録音中は Enter でも停止できます\n"
            "灰(ロード中)でも録音は始められます\n"
            "右クリック = メニュー"
        )
        self.resize(BTN_SIZE, BTN_SIZE)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - BTN_SIZE - 40, screen.bottom() - BTN_SIZE - 80)

        self.state_changed.connect(self._on_state)
        self.preview_text.connect(self._on_preview)
        self.notify.connect(self._on_notify)

        self.show()
        apply_noactivate(self)

        threading.Thread(target=self._load_models, daemon=True).start()

        self._hk = pynput_keyboard.GlobalHotKeys({
            HOTKEY: lambda: self.notify.emit("__toggle__"),
            WAKE_HOTKEY: lambda: self.notify.emit("__wake_toggle__"),
        })
        self._hk.start()

        self._enter_suppressed = False
        self._enter_hook_proc = _HOOKPROC(self._on_low_level_key)
        self._enter_hook_id = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._enter_hook_proc, None, 0)
        QApplication.instance().aboutToQuit.connect(self._uninstall_enter_hook)
        QApplication.instance().aboutToQuit.connect(self._stop_listening)

    def _load_models(self):
        # 確定用(large-v3)を先にロードする。録音はモデル無しで始められるので、
        # ユーザーが最初に待たされるのは「停止 → 確定」であり、そこに要るのはこちら。
        # 暫定プレビュー(small)は無くても録音は進むので後回しでよい。
        self.core.load_model()
        self.core.warmup()
        self._final_ready.set()
        self.core.load_preview_model(PREVIEW_MODEL)
        self.core.warmup()
        # VAD も先に読み込む(音声が流れ始めてから引くとコールバックが詰まる)
        self._wake.preload()
        self._dictate.preload()
        self._models_ready.set()
        # 状態遷移はメインスレッドで判断する(ロード中に録音が始まっていれば
        # 今は RECORDING/BUSY なので、ここで IDLE に上書きしてはいけない)
        self.notify.emit("__models_ready__")

    # ---- ウェイクワード待受 ----
    def set_wake_mode(self, on):
        if on and self.state == LOADING:
            safe_log("[wake] モデルのロード中は待受を開始できません")
            return
        self.wake_on = bool(on)
        if self.wake_on:
            self._start_listening()
        elif self.state == SESSION:
            self._end_session()   # 口述中に OFF にされたらセッションごと畳む
        else:
            self._stop_listening()
        safe_log(f"[wake] 待受 {'ON' if self.wake_on else 'OFF'}")

    def _start_listening(self):
        """IDLE のときだけマイク監視を開く(録音中は録音側がマイクを使う)。"""
        if not self.wake_on or self.state != IDLE:
            return
        try:
            self._wake.start()
            self.core.start_monitor(self._wake.feed)
        except Exception as e:
            safe_log(f"[wake] マイク監視を開始できませんでした: {e}")
            self._wake.stop()
            self.wake_on = False
        self.update()

    def _stop_listening(self):
        self.core.stop_monitor()
        self._wake.stop()
        self.update()

    def _wake_transcribe(self, audio):
        with self._model_lock:
            return self.core.transcribe_preview(audio)

    # ---- 連続口述セッション ----
    def _start_session(self):
        """ウェイクワード検出後、黙るまでを1発話として貼り続けるモードに入る。

        マイクは待受のまま使い続ける(閉じて開き直さないので発話の頭が欠けない)。
        渡し先だけウェイク判定 → 口述に差し替える。
        """
        self._wake.stop()
        self._dictate.start()
        self.core.start_monitor(self._dictate.feed)
        self._show_preview("聞いています…")
        self.state_changed.emit(SESSION)
        threading.Thread(target=self._session_watchdog, daemon=True).start()
        safe_log("[session] 開始: 喋る → 黙る → 貼り付け。止めるにはボタン/Ctrl+Alt+Space")

    def _end_session(self):
        self._dictate.stop()
        self.preview.hide()
        if not self.wake_on:
            self.core.stop_monitor()
        safe_log("[session] 終了")
        self.state_changed.emit(IDLE)  # 待受 ON なら _on_state が待受を再開する

    def _on_utterance(self, audio):
        """1発話ぶんの音声を確定して貼り付ける(SpeechSegmenter のスレッドで直列実行)。"""
        t_start = time.time()
        with self._model_lock:
            text = self.core.transcribe(audio)
        t_text = time.time()
        if text:
            self.preview_text.emit(text)
            self.core.deliver(text)
        else:
            self.preview_text.emit("(認識できませんでした)")
        # 「喋り終わってから貼られるまで」の体感 = DICTATE_SILENCE_SEC + 確定 + 貼付。
        # 貼付にクリップボード復元待ちは含まない(別スレッドへ回してあるため)。
        safe_log(f"[timing] 音声 {len(audio) / SAMPLE_RATE:.1f}s / "
                 f"確定 {t_text - t_start:.2f}s / 貼付 {time.time() - t_text:.2f}s / "
                 f"体感 {DICTATE_SILENCE_SEC + time.time() - t_start:.2f}s")

    def _session_watchdog(self):
        """しばらく何も喋らなければセッションを終える(言いっぱなしの放置対策)。"""
        while self.state == SESSION:
            time.sleep(0.5)
            idle = time.monotonic() - self._dictate.last_speech_at
            if idle >= SESSION_IDLE_TIMEOUT_SEC:
                self.notify.emit("__end_session__")
                return

    # ---- 録音トグル ----
    def toggle(self):
        if self.state == SESSION:
            self._end_session()
        elif self.state in (IDLE, LOADING):
            # LOADING でも録音は始める。マイクにモデルは不要で、確定に要る
            # large-v3 は _finish で待つ(録音している間にロードが進む)
            self._stop_listening()   # 録音と待受で同じマイクを取り合わないよう先に閉じる
            self.core.start_recording()
            self.state_changed.emit(RECORDING)
            self._start_preview()
        elif self.state == RECORDING:
            self._preview_running = False
            self.state_changed.emit(BUSY)
            threading.Thread(target=self._finish, daemon=True).start()

    def _show_preview(self, text=""):
        self.preview.set_text(text)
        if not self.preview.user_moved:
            self.preview.place_above(self.frameGeometry())
        self.preview.show()

    def _start_preview(self):
        self._show_preview("" if self.core.preview_model is not None
                           else "(モデル読込中… 録音は続いています)")
        self._preview_running = True
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def _preview_loop(self):
        while self._preview_running:
            t0 = time.time()
            audio = self.core.snapshot_audio()
            if self.core.preview_model is None:
                audio = None   # 暫定モデルがまだ無い間は待つだけ(録音は進んでいる)
            if audio is not None and len(audio) > SAMPLE_RATE * 0.3:
                try:
                    with self._model_lock:
                        text = self.core.transcribe_preview(audio)
                    if self._preview_running:
                        self.preview_text.emit(text)
                except Exception:
                    pass
            # 変換にかかった時間を差し引いて待つ(速ければ高頻度、遅ければ詰まらない)
            time.sleep(max(0.05, PREVIEW_INTERVAL - (time.time() - t0)))

    def _finish(self):
        try:
            audio = self.core.stop_recording()
            if audio is None or len(audio) == 0:
                return
            if not self._final_ready.is_set():
                # 起動直後に喋り始めたケース。ここで待つのは残りのロード時間だけ
                self.preview_text.emit("(モデル読込中… そのまま待ってください)")
                self._final_ready.wait()
            t_stop = time.time()
            text = self.core.transcribe(audio)
            t_text = time.time()
            if text:
                self.core.deliver(text)
            # 遅く感じたときにどこが重いかを当てずっぽうでなく数字で見るためのログ。
            # 「喋り終わり → 貼り付け」の体感 = AUTO_STOP_SILENCE_SEC + 確定 + 貼付。
            safe_log(f"[timing] 音声 {len(audio) / SAMPLE_RATE:.1f}s / "
                  f"確定 {t_text - t_stop:.2f}s / 貼付 {time.time() - t_text:.2f}s")
        finally:
            self.preview_text.emit("__hide__")
            # 全モデルが揃う前に録音していたなら灰に戻す(揃えば IDLE)。
            # 待受 ON なら _on_state がマイク監視を再開する
            self.state_changed.emit(IDLE if self._models_ready.is_set() else LOADING)

    # ---- 録音中の Enter キーで停止(低レベルフック、メインスレッドで呼ばれる) ----
    def _on_low_level_key(self, nCode, wParam, lParam):
        # このコールバックは「システム全体のキー入力」が通る経路。ここで例外が出ると
        # CallNextHookEx が呼ばれないまま 0 が返り、そのキーイベントがフックチェーンで
        # 止まる = IME・支援技術・他のホットキー管理ツールがそのキーを取り逃す。
        # 自分が壊れても他アプリの入力経路は壊さないため、全体を try で包む。
        # (終了処理中に Qt オブジェクトが破棄済みで self.state アクセスが
        #  RuntimeError になるケースが実際に起こりうる)
        try:
            if nCode == 0:
                kb = ctypes.cast(lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                if kb.vkCode == VK_RETURN:
                    if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                        if self._enter_suppressed:
                            return 1  # キーリピート中: 抑制だけ継続し toggle は再発火させない
                        if self.state == RECORDING:
                            self._enter_suppressed = True
                            self.notify.emit("__toggle__")
                            return 1
                    elif wParam in (WM_KEYUP, WM_SYSKEYUP):
                        if self._enter_suppressed:
                            self._enter_suppressed = False
                            return 1
        except Exception:
            pass  # 握り潰してでも下の CallNextHookEx へ必ず到達させる
        return user32.CallNextHookEx(self._enter_hook_id, nCode, wParam, lParam)

    def _uninstall_enter_hook(self):
        if getattr(self, "_enter_hook_id", None):
            user32.UnhookWindowsHookEx(self._enter_hook_id)
            self._enter_hook_id = None

    # ---- シグナルハンドラ(メインスレッド) ----
    def _on_state(self, state):
        if state == LOADING and self._models_ready.is_set():
            state = IDLE   # ロード完了と _finish の「灰へ戻す」が競合したときの保険
        self.state = state
        if state == IDLE and self.wake_on:
            self._start_listening()
        self.update()

    def _on_preview(self, text):
        if text == "__hide__":
            self.preview.hide()
        else:
            self.preview.set_text(text)

    def _on_notify(self, msg):
        if msg == "__toggle__":
            self.toggle()
        elif msg == "__wake_toggle__":
            self.set_wake_mode(not self.wake_on)
        elif msg == "__wake__":
            # ウェイクワード検出。この時点で待受は自分で止まっている。
            if self.state == IDLE:
                self._start_session()
            else:
                self._start_listening()  # 録音中などに来たら無視して待受へ戻す
        elif msg == "__end_session__":
            if self.state == SESSION:
                self._end_session()
        elif msg == "__models_ready__":
            if self.state == LOADING:
                self._on_state(IDLE)   # 録音中なら _finish 側が IDLE に戻す

    # ---- 描画 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = COLORS.get(self.state, COLORS[IDLE])
        if self.state == IDLE and self.wake_on:
            color = COLORS[LISTENING]   # 緑 = ウェイクワードを待っている
        p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        m = 6
        p.drawEllipse(m, m, self.width() - 2 * m, self.height() - 2 * m)

        cx, cy = self.width() // 2, self.height() // 2
        if self.state in (IDLE, SESSION):
            # SESSION は赤地にマイク = 「今喋ったら貼られる」状態
            self._draw_mic(p, cx, cy)
        else:
            p.setPen(QColor("white"))
            f = QFont()
            f.setPointSize(18)
            f.setBold(True)
            p.setFont(f)
            sym = {LOADING: "…", RECORDING: "■", BUSY: "…"}.get(self.state, "")
            p.drawText(self.rect(), Qt.AlignCenter, sym)
        p.end()

    def _draw_mic(self, p, cx, cy):
        p.setBrush(QBrush(QColor("white")))
        p.setPen(Qt.NoPen)
        w, h = 13, 20
        p.drawRoundedRect(cx - w // 2, cy - h // 2 - 3, w, h, 6, 6)
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(cx, cy + h // 2 - 1, cx, cy + h // 2 + 6)
        p.drawLine(cx - 7, cy + h // 2 + 6, cx + 7, cy + h // 2 + 6)

    # ---- ドラッグ移動 & クリック ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._moved = False
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and (e.buttons() & Qt.LeftButton):
            new = e.globalPosition().toPoint() - self._drag_pos
            if (new - self.frameGeometry().topLeft()).manhattanLength() > 3:
                self._moved = True
            self.move(new)
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if not self._moved:
                self.toggle()   # クリックで録音 開始/停止(生口述)
            self._drag_pos = None
            e.accept()
        elif e.button() == Qt.RightButton:
            m = QMenu()
            wake = m.addAction("ウェイクワード待受(「やっほークロード」)")
            wake.setCheckable(True)
            wake.setChecked(self.wake_on)
            wake.setEnabled(self.state != LOADING)
            wake.triggered.connect(self.set_wake_mode)
            m.addSeparator()
            m.addAction("終了").triggered.connect(QApplication.quit)
            m.exec(e.globalPosition().toPoint())


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    MicButton()

    tray = QSystemTrayIcon()
    pix = QPixmap(32, 32)
    pix.fill(QColor("#2d7dd2"))
    tray.setIcon(QIcon(pix))
    tray.setToolTip("voice-typing (realtime)")
    menu = QMenu()
    menu.addAction("終了").triggered.connect(app.quit)
    tray.setContextMenu(menu)
    tray.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
