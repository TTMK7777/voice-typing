"""タスクバー/ショートカット用のマイクアイコン(icon.ico)を生成する。"""
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter, QColor, QBrush, QPen
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
S = 256
pix = QPixmap(S, S)
pix.fill(Qt.transparent)
p = QPainter(pix)
p.setRenderHint(QPainter.Antialiasing)

# 背景円
p.setBrush(QBrush(QColor("#2d7dd2")))
p.setPen(Qt.NoPen)
p.drawEllipse(8, 8, S - 16, S - 16)

# マイク本体(白)
cx, cy = S // 2, S // 2
p.setBrush(QBrush(QColor("white")))
w, h = 60, 96
p.drawRoundedRect(cx - w // 2, cy - h // 2 - 14, w, h, 28, 28)

# スタンド
pen = QPen(QColor("white"))
pen.setWidth(10)
pen.setCapStyle(Qt.RoundCap)
p.setPen(pen)
p.drawLine(cx, cy + h // 2 - 6, cx, cy + h // 2 + 26)
p.drawLine(cx - 34, cy + h // 2 + 26, cx + 34, cy + h // 2 + 26)

# 受け(アーク)
p.setBrush(Qt.NoBrush)
p.drawArc(cx - 44, cy - 24, 88, 104, 200 * 16, 140 * 16)
p.end()

ok = pix.save("icon.ico", "ICO")
print("icon.ico saved:", ok)
