"""
リアルタイム版: フローティングのマイクボタン + 暫定テキストのプレビュー。

- 録音中、small モデルで PREVIEW_INTERVAL 秒ごとに暫定テキストを更新表示(揺れる)
- 停止で large-v3 が高精度確定 → アクティブウィンドウに自動入力
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

IDLE, LOADING, RECORDING, BUSY = "idle", "loading", "recording", "busy"
COLORS = {
    IDLE: QColor("#2d7dd2"),
    LOADING: QColor("#868e96"),
    RECORDING: QColor("#e03131"),
    BUSY: QColor("#f08c00"),
}
HOTKEY = "<ctrl>+<alt>+<space>"        # 生口述(録音 開始/停止)
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

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setToolTip(
            "クリック / Ctrl+Alt+Space = 録音 開始/停止\n"
            "録音中は Enter でも停止できます\n"
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
        })
        self._hk.start()

        self._enter_suppressed = False
        self._enter_hook_proc = _HOOKPROC(self._on_low_level_key)
        self._enter_hook_id = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._enter_hook_proc, None, 0)
        QApplication.instance().aboutToQuit.connect(self._uninstall_enter_hook)

    def _load_models(self):
        # 暫定用(small)→ 確定用(large-v3)の順でロード
        self.core.load_preview_model(PREVIEW_MODEL)
        self.core.load_model()
        self.state_changed.emit(IDLE)

    # ---- 録音トグル ----
    def toggle(self):
        if self.state == LOADING:
            return
        if self.state == IDLE:
            self.core.start_recording()
            self.state_changed.emit(RECORDING)
            self._start_preview()
        elif self.state == RECORDING:
            self._preview_running = False
            self.state_changed.emit(BUSY)
            threading.Thread(target=self._finish, daemon=True).start()

    def _start_preview(self):
        self.preview.set_text("")
        if not self.preview.user_moved:
            self.preview.place_above(self.frameGeometry())
        self.preview.show()
        self._preview_running = True
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def _preview_loop(self):
        while self._preview_running:
            t0 = time.time()
            audio = self.core.snapshot_audio()
            if audio is not None and len(audio) > SAMPLE_RATE * 0.3:
                try:
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
            text = self.core.transcribe(audio)
            if text:
                self.core.deliver(text)
        finally:
            self.preview_text.emit("__hide__")
            self.state_changed.emit(IDLE)

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
        self.state = state
        self.update()

    def _on_preview(self, text):
        if text == "__hide__":
            self.preview.hide()
        else:
            self.preview.set_text(text)

    def _on_notify(self, msg):
        if msg == "__toggle__":
            self.toggle()

    # ---- 描画 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(COLORS.get(self.state, COLORS[IDLE])))
        p.setPen(Qt.NoPen)
        m = 6
        p.drawEllipse(m, m, self.width() - 2 * m, self.height() - 2 * m)

        cx, cy = self.width() // 2, self.height() // 2
        if self.state == IDLE:
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
