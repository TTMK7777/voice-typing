"""
フローティングのマイクボタン常駐GUI。
押す(orホットキー)→録音→今アクティブなウィンドウに自動入力。

- 最前面に常駐、ドラッグで移動可
- フォーカスを奪わない(WS_EX_NOACTIVATE)ので入力先がズレない
- 状態を色で表示: 灰=ロード中 / 青=待機 / 赤=録音中 / 橙=変換中
- 右クリック or トレイから終了
"""
import sys
import threading
import ctypes

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QFont, QPen, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QSystemTrayIcon, QMenu
from pynput import keyboard as pynput_keyboard

from core import VoiceCore

IDLE, LOADING, RECORDING, BUSY = "idle", "loading", "recording", "busy"
COLORS = {
    IDLE: QColor("#2d7dd2"),
    LOADING: QColor("#868e96"),
    RECORDING: QColor("#e03131"),
    BUSY: QColor("#f08c00"),
}
HOTKEY = "<ctrl>+<alt>+<space>"
BTN_SIZE = 64


class MicButton(QWidget):
    # ワーカースレッド → UI更新用シグナル(emit はスレッドセーフ)
    state_changed = Signal(str)
    notify = Signal(str)

    def __init__(self):
        super().__init__()
        self.state = LOADING
        self.core = VoiceCore()
        self._drag_pos = None
        self._moved = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setToolTip("クリック or Ctrl+Alt+Space で録音 / 右クリックで終了")
        self.resize(BTN_SIZE, BTN_SIZE)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - BTN_SIZE - 40, screen.bottom() - BTN_SIZE - 80)

        self.state_changed.connect(self._on_state)
        self.notify.connect(self._on_notify)

        self.show()
        self._apply_noactivate()

        threading.Thread(target=self._load_model, daemon=True).start()

        self._hk = pynput_keyboard.GlobalHotKeys({HOTKEY: self._hotkey_toggle})
        self._hk.start()

    def _apply_noactivate(self):
        """フォーカスを奪わないウィンドウにする(Windows)。これで入力先が保持される。"""
        try:
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        except Exception:
            pass

    def _load_model(self):
        self.core.load_model()
        self.state_changed.emit(IDLE)

    # ---- 録音トグル ----
    def _hotkey_toggle(self):
        # pynput スレッドから → シグナル経由でメインスレッドへ受け渡し
        self.notify.emit("__toggle__")

    def toggle(self):
        if self.state == LOADING:
            return
        if self.state == IDLE:
            self.core.start_recording()
            self.state_changed.emit(RECORDING)
        elif self.state == RECORDING:
            self.state_changed.emit(BUSY)
            threading.Thread(target=self._finish, daemon=True).start()

    def _finish(self):
        try:
            audio = self.core.stop_recording()
            if audio is None or len(audio) == 0:
                self.state_changed.emit(IDLE)
                return
            text = self.core.transcribe(audio)
            if text:
                self.core.deliver(text)
        finally:
            self.state_changed.emit(IDLE)

    # ---- シグナルハンドラ(メインスレッド) ----
    def _on_state(self, state):
        self.state = state
        self.update()

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
        """白いマイクのアイコンを描く(絵文字フォント非依存)。"""
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
                self.toggle()
            self._drag_pos = None
            e.accept()
        elif e.button() == Qt.RightButton:
            self._show_menu(e.globalPosition().toPoint())

    def _show_menu(self, pos):
        m = QMenu()
        m.addAction("終了").triggered.connect(QApplication.quit)
        m.exec(pos)


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    MicButton()

    tray = QSystemTrayIcon()
    pix = QPixmap(32, 32)
    pix.fill(QColor("#2d7dd2"))
    tray.setIcon(QIcon(pix))
    tray.setToolTip("voice-typing")
    menu = QMenu()
    menu.addAction("終了").triggered.connect(app.quit)
    tray.setContextMenu(menu)
    tray.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
