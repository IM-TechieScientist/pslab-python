#!/usr/bin/env python3
"""PSLab Pico waveform workbench.

This is a development GUI for the RP2350 PSLab firmware. It is intentionally
standalone under tools/ so it can evolve quickly while the firmware protocol is
still moving.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from serial.tools import list_ports

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit(
        "Missing GUI dependency. Install PyQt5, then run again.\n"
        "Example: python3 -m pip install PyQt5"
    ) from exc

from pslab.instrument.pico_logic_analyzer import LogicCapture, PicoLogicAnalyzer
from pslab.instrument.pico_oscilloscope import PicoOscilloscope, ScopeCapture


FRAME_LEN = 512
OUTER_HEADER_LEN = 12
OUTER_PAYLOAD_LEN = FRAME_LEN - OUTER_HEADER_LEN
OUTER_MAGIC = 0xA5
OUTER_TYPE_DATA = 0x02
PSLAB_MAGIC = b"PSLB"
PSLAB_HEADER_LEN = 32
PSLAB_SUBTYPE_META = 1
PSLAB_SUBTYPE_DATA = 2
CRASH_LOG = Path.home() / "pslab_pico_waveform_workbench_crash.log"


@dataclass
class WifiMeta:
    instrument: int
    capture_sequence: int
    sample_rate: int
    sample_count: int
    channel_count: int
    source: int
    trigger_mode: int
    data_format: int


@dataclass
class WifiCapture:
    meta: WifiMeta
    payload: bytes


def checksum8(data: bytes) -> int:
    return sum(data) & 0xFF


def available_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def log_exception(prefix: str) -> str:
    text = f"{prefix}\n{traceback.format_exc()}"
    try:
        CRASH_LOG.write_text(text)
    except OSError:
        pass
    return text


class WaveformView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QtGui.QColor("#101318"))
        self.base_x_scale = 0.08
        self.x_scale = self.base_x_scale
        self.autoscroll = False
        self.show_grid = True
        self.channel_visible: dict[str, bool] = {}
        self._last_duration_us = 1.0
        self.preserve_sequence_gaps = False
        self.max_render_points = 18000

    def set_autoscroll(self, enabled: bool) -> None:
        self.autoscroll = enabled

    def set_channel_visible(self, channel: str, visible: bool) -> None:
        self.channel_visible[channel] = visible

    def set_preserve_sequence_gaps(self, enabled: bool) -> None:
        self.preserve_sequence_gaps = enabled

    def scroll_to_latest(self) -> None:
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().maximum())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.ControlModifier:
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self.scale(factor, 1.0)
            return
        super().wheelEvent(event)

    def zoom_in(self) -> None:
        self.scale(1.25, 1.0)

    def zoom_out(self) -> None:
        self.scale(0.8, 1.0)

    def fit_time(self) -> None:
        rect = self.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def _reset_scene(
        self,
        title: str,
        duration_us: float,
        lane_count: int,
        preserve_scale: bool = False,
        time_origin_us: float = 0.0,
    ) -> None:
        self.scene().clear()
        self._last_duration_us = max(duration_us, 0.001)
        if not preserve_scale:
            target_plot_width = max(float(self.viewport().width()) - 180.0, 900.0)
            self.x_scale = max(self.base_x_scale, target_plot_width / self._last_duration_us)
        width = max(1200.0, self._last_duration_us * self.x_scale + 220.0)
        height = max(360.0, lane_count * 86.0 + 80.0)
        self.setSceneRect(0, 0, width, height)

        title_item = self.scene().addText(title, QtGui.QFont("Inter", 12, QtGui.QFont.Bold))
        title_item.setDefaultTextColor(QtGui.QColor("#e8edf2"))
        title_item.setPos(12, 8)

        if self.show_grid:
            self._draw_grid(width, height, duration_us, time_origin_us)

    def _draw_grid(self, width: float, height: float, duration_us: float, time_origin_us: float = 0.0) -> None:
        grid_pen = QtGui.QPen(QtGui.QColor("#252b33"))
        grid_pen.setWidth(0)
        axis_pen = QtGui.QPen(QtGui.QColor("#3b4350"))
        axis_pen.setWidth(0)
        left = 96.0
        top = 44.0
        self.scene().addLine(left, top, left, height - 22, axis_pen)
        self.scene().addLine(left, height - 34, width - 20, height - 34, axis_pen)

        target_grid_px = 120
        step_us = max(duration_us / max((width - left) / target_grid_px, 1.0), 0.001)
        nice_steps = np.array([
            0.001, 0.002, 0.005,
            0.01, 0.02, 0.05,
            0.1, 0.2, 0.5,
            1, 2, 5,
            10, 20, 50,
            100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000,
            100000, 200000, 500000,
        ])
        step_us = float(nice_steps[np.argmin(np.abs(nice_steps - step_us))])

        t = 0.0
        font = QtGui.QFont("Inter", 8)
        while t <= duration_us:
            x = left + t * self.x_scale
            self.scene().addLine(x, top, x, height - 34, grid_pen)
            label = self.scene().addText(self._format_time_label(time_origin_us + t), font)
            label.setDefaultTextColor(QtGui.QColor("#9aa7b5"))
            label.setPos(x + 4, height - 32)
            t += step_us

    @staticmethod
    def _format_time_label(time_us: float) -> str:
        if time_us < 1.0:
            return f"{time_us * 1000:g} ns"
        if time_us < 1000.0:
            return f"{time_us:g} us"
        return f"{time_us / 1000:g} ms"

    def plot_logic(self, capture: LogicCapture, sequence: int | None = None) -> None:
        self.plot_logic_history([(sequence, capture)])

    def plot_logic_history(self, captures: list[tuple[int | None, LogicCapture]]) -> None:
        if not captures:
            return

        latest = captures[-1][1]
        offsets, time_origin_us = self._history_offsets(captures)
        duration_us = 0.0
        if captures:
            duration_us = offsets[-1] + captures[-1][1].samples * captures[-1][1].sample_interval_us
        first_seq = captures[0][0]
        last_seq = captures[-1][0]
        seq_text = (
            f"{first_seq}..{last_seq}"
            if first_seq is not None and last_seq is not None and first_seq != last_seq
            else f"{last_seq if last_seq is not None else '-'}"
        )
        title = (
            f"Logic Analyzer"
            f"  seq={seq_text}"
            f"  {latest.sample_rate / 1e6:.3f} MS/s"
            f"  window={duration_us / 1000:.2f} ms"
        )
        visible_channels = [
            i for i in range(latest.pin_count)
            if self.channel_visible.get(f"GPIO{latest.pin_base + i}", True)
        ]
        self._reset_scene(
            title,
            duration_us,
            max(len(visible_channels), 1),
            preserve_scale=len(captures) > 1,
            time_origin_us=time_origin_us,
        )

        left = 96.0
        lane_h = 78.0
        top = 58.0
        colors = ["#49a6ff", "#f6c350", "#4cd964", "#ff6b6b", "#b073ff", "#21d4a4", "#ff9f43", "#d9e2ec"]
        font = QtGui.QFont("Inter", 9)
        pen_zero = QtGui.QPen(QtGui.QColor("#2e3641"))
        pen_zero.setWidth(0)

        for lane, pin_index in enumerate(visible_channels):
            gpio = latest.pin_base + pin_index
            y0 = top + lane * lane_h + 44
            y1 = y0 - 34
            self.scene().addLine(left, y0, self.sceneRect().width() - 20, y0, pen_zero)

            label = self.scene().addText(f"GPIO{gpio}", font)
            label.setDefaultTextColor(QtGui.QColor("#d5dde5"))
            label.setPos(18, y1 - 8)

            path = QtGui.QPainterPath()
            path_started = False
            render_stride = self._logic_render_stride(captures, max(len(visible_channels), 1))
            for entry, time_offset in zip(captures, offsets):
                capture = entry[1]
                if pin_index >= capture.pin_count:
                    continue
                states = capture.states[pin_index]
                if states.size == 0:
                    continue

                x = left + time_offset * self.x_scale
                y_prev = y1 if states[0] else y0
                if not path_started:
                    path.moveTo(x, y_prev)
                    path_started = True
                else:
                    path.lineTo(x, y_prev)

                for sample_index in range(render_stride, capture.samples, render_stride):
                    x = left + (time_offset + sample_index * capture.sample_interval_us) * self.x_scale
                    y = y1 if states[sample_index] else y0
                    path.lineTo(x, y_prev)
                    path.lineTo(x, y)
                    y_prev = y

                end_x = left + (time_offset + capture.samples * capture.sample_interval_us) * self.x_scale
                path.lineTo(end_x, y_prev)

            pen = QtGui.QPen(QtGui.QColor(colors[pin_index % len(colors)]))
            pen.setWidth(2)
            self.scene().addPath(path, pen)

        if self.autoscroll:
            QtCore.QTimer.singleShot(0, self.scroll_to_latest)

    def plot_logic_single_old(self, capture: LogicCapture, sequence: int | None = None) -> None:
        duration_us = max((capture.samples - 1) * capture.sample_interval_us, capture.sample_interval_us)
        title = (
            f"Logic Analyzer"
            f"  seq={sequence if sequence is not None else '-'}"
            f"  {capture.sample_rate / 1e6:.3f} MS/s"
            f"  {capture.samples} samples"
        )
        visible_channels = [
            i for i in range(capture.pin_count)
            if self.channel_visible.get(f"GPIO{capture.pin_base + i}", True)
        ]
        self._reset_scene(title, duration_us, max(len(visible_channels), 1))

        left = 96.0
        lane_h = 78.0
        top = 58.0
        colors = ["#49a6ff", "#f6c350", "#4cd964", "#ff6b6b", "#b073ff", "#21d4a4", "#ff9f43", "#d9e2ec"]
        font = QtGui.QFont("Inter", 9)
        pen_zero = QtGui.QPen(QtGui.QColor("#2e3641"))
        pen_zero.setWidth(0)

        for lane, pin_index in enumerate(visible_channels):
            gpio = capture.pin_base + pin_index
            y0 = top + lane * lane_h + 44
            y1 = y0 - 34
            self.scene().addLine(left, y0, self.sceneRect().width() - 20, y0, pen_zero)

            label = self.scene().addText(f"GPIO{gpio}", font)
            label.setDefaultTextColor(QtGui.QColor("#d5dde5"))
            label.setPos(18, y1 - 8)

            path = QtGui.QPainterPath()
            states = capture.states[pin_index]
            x_prev = left
            y_prev = y1 if states[0] else y0
            path.moveTo(x_prev, y_prev)
            for sample_index in range(1, capture.samples):
                x = left + sample_index * capture.sample_interval_us * self.x_scale
                y = y1 if states[sample_index] else y0
                path.lineTo(x, y_prev)
                path.lineTo(x, y)
                x_prev = x
                y_prev = y
            path.lineTo(left + duration_us * self.x_scale, y_prev)

            pen = QtGui.QPen(QtGui.QColor(colors[pin_index % len(colors)]))
            pen.setWidth(2)
            self.scene().addPath(path, pen)

        if self.autoscroll:
            QtCore.QTimer.singleShot(0, self.scroll_to_latest)

    def plot_scope(self, capture: ScopeCapture, sequence: int | None = None) -> None:
        self.plot_scope_history([(sequence, capture)])

    def plot_scope_history(self, captures: list[tuple[int | None, ScopeCapture]]) -> None:
        if not captures:
            return

        latest = captures[-1][1]
        offsets, time_origin_us = self._history_offsets(captures)
        duration_us = 0.0
        if captures:
            duration_us = offsets[-1] + captures[-1][1].samples * captures[-1][1].sample_interval_us
        first_seq = captures[0][0]
        last_seq = captures[-1][0]
        seq_text = (
            f"{first_seq}..{last_seq}"
            if first_seq is not None and last_seq is not None and first_seq != last_seq
            else f"{last_seq if last_seq is not None else '-'}"
        )
        title = (
            f"Oscilloscope"
            f"  seq={seq_text}"
            f"  GPIO{latest.gpio}"
            f"  {latest.sample_rate / 1000:.1f} kS/s"
            f"  window={duration_us / 1000:.2f} ms"
        )
        self._reset_scene(title, duration_us, 1, preserve_scale=len(captures) > 1, time_origin_us=time_origin_us)

        left = 96.0
        top = 66.0
        graph_h = 250.0
        pen_grid = QtGui.QPen(QtGui.QColor("#2e3641"))
        pen_grid.setWidth(0)
        for fraction, label_text in [(0.0, "3.3 V"), (0.5, "1.65 V"), (1.0, "0 V")]:
            y = top + graph_h * fraction
            self.scene().addLine(left, y, self.sceneRect().width() - 20, y, pen_grid)
            label = self.scene().addText(label_text, QtGui.QFont("Inter", 9))
            label.setDefaultTextColor(QtGui.QColor("#d5dde5"))
            label.setPos(22, y - 10)

        path = QtGui.QPainterPath()
        path_started = False
        render_stride = self._scope_render_stride(captures)
        for entry, time_offset in zip(captures, offsets):
            capture = entry[1]
            volts = capture.volts
            for i in range(0, len(volts), render_stride):
                value = volts[i]
                x = left + (time_offset + i * capture.sample_interval_us) * self.x_scale
                y = top + graph_h - (float(value) / 3.3) * graph_h
                if not path_started:
                    path.moveTo(x, y)
                    path_started = True
                else:
                    path.lineTo(x, y)

        pen = QtGui.QPen(QtGui.QColor("#49a6ff"))
        pen.setWidth(2)
        self.scene().addPath(path, pen)

        if self.autoscroll:
            QtCore.QTimer.singleShot(0, self.scroll_to_latest)

    def _history_offsets(self, captures):
        if captures and len(captures[0]) >= 3:
            absolute_offsets = [float(entry[2]) for entry in captures]
            origin = absolute_offsets[0]
            return [offset - origin for offset in absolute_offsets], origin

        offsets = []
        current = 0.0
        previous_sequence = None
        previous_duration = 0.0
        for sequence, capture in captures:
            if previous_sequence is not None:
                step = previous_duration
                if (
                    self.preserve_sequence_gaps
                    and sequence is not None
                    and previous_sequence is not None
                    and sequence > previous_sequence + 1
                ):
                    step *= min(sequence - previous_sequence, 20)
                current += step
            offsets.append(current)
            previous_sequence = sequence
            previous_duration = capture.samples * capture.sample_interval_us
        return offsets, 0.0

    def _logic_render_stride(self, captures, visible_channel_count: int) -> int:
        total_points = sum(max(0, entry[1].samples) for entry in captures) * max(visible_channel_count, 1)
        return max(1, int(np.ceil(total_points / max(self.max_render_points, 1))))

    def _scope_render_stride(self, captures) -> int:
        total_points = sum(max(0, entry[1].samples) for entry in captures)
        return max(1, int(np.ceil(total_points / max(self.max_render_points, 1))))

    def plot_scope_single_old(self, capture: ScopeCapture, sequence: int | None = None) -> None:
        duration_us = max((capture.samples - 1) * capture.sample_interval_us, capture.sample_interval_us)
        title = (
            f"Oscilloscope"
            f"  seq={sequence if sequence is not None else '-'}"
            f"  GPIO{capture.gpio}"
            f"  {capture.sample_rate / 1000:.1f} kS/s"
            f"  {capture.samples} samples"
        )
        self._reset_scene(title, duration_us, 1)

        left = 96.0
        top = 66.0
        graph_h = 250.0
        mid = top + graph_h / 2
        pen_grid = QtGui.QPen(QtGui.QColor("#2e3641"))
        pen_grid.setWidth(0)
        for fraction, label_text in [(0.0, "3.3 V"), (0.5, "1.65 V"), (1.0, "0 V")]:
            y = top + graph_h * fraction
            self.scene().addLine(left, y, self.sceneRect().width() - 20, y, pen_grid)
            label = self.scene().addText(label_text, QtGui.QFont("Inter", 9))
            label.setDefaultTextColor(QtGui.QColor("#d5dde5"))
            label.setPos(22, y - 10)

        volts = capture.volts
        path = QtGui.QPainterPath()
        path.moveTo(left, mid)
        for i, value in enumerate(volts):
            x = left + i * capture.sample_interval_us * self.x_scale
            y = top + graph_h - (float(value) / 3.3) * graph_h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pen = QtGui.QPen(QtGui.QColor("#49a6ff"))
        pen.setWidth(2)
        self.scene().addPath(path, pen)

        if self.autoscroll:
            QtCore.QTimer.singleShot(0, self.scroll_to_latest)


class CaptureWorker(QtCore.QThread):
    capture_ready = QtCore.pyqtSignal(object, int, str)
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    finished_cleanly = QtCore.pyqtSignal()

    def __init__(self, instrument: str, port: str | None, config: dict, repeat: bool):
        super().__init__()
        self.instrument = instrument
        self.port = port
        self.config = config
        self.repeat = repeat
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        sequence = 0
        try:
            while not self._stop:
                if self.instrument == "LA":
                    device = PicoLogicAnalyzer(port=self.port, timeout=2.0, sys_clock_hz=self.config["sys_clock_hz"])
                    capture = device.capture(
                        pin_base=self.config["pin_base"],
                        pin_count=self.config["pin_count"],
                        samples=self.config["samples"],
                        divider=self.config["divider"],
                        trigger_pin=self.config["trigger_pin"],
                        trigger_level=self.config["trigger_level"],
                        trigger_mode=self.config["trigger_mode"],
                    )
                    device.disconnect()
                    self.capture_ready.emit(capture, sequence, "LA")
                else:
                    device = PicoOscilloscope(port=self.port, timeout=2.0)
                    capture = device.capture(
                        channel=self.config["dso_channel"],
                        sample_rate=self.config["dso_rate"],
                        samples=self.config["dso_samples"],
                        trigger_level=self.config["dso_trigger_level"],
                        trigger_mode=self.config["dso_trigger_mode"],
                        trigger_slope=self.config["dso_trigger_slope"],
                    )
                    device.disconnect()
                    self.capture_ready.emit(capture, sequence, "DSO")

                sequence += 1
                if not self.repeat:
                    break
                self.msleep(max(10, int(self.config.get("repeat_delay_ms", 20))))
        except Exception as exc:  # pragma: no cover - UI path
            self.failed.emit(str(exc))
        finally:
            self.finished_cleanly.emit()


class WifiReceiver(QtCore.QThread):
    capture_ready = QtCore.pyqtSignal(object, int, str)
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, port: int = 5005, esp_host: str | None = None):
        super().__init__()
        self.port = port
        self.esp_host = esp_host
        self._stop = False
        self._meta: WifiMeta | None = None
        self._chunks: dict[int, bytes] = {}
        self._chunk_count = 0
        self.max_display_hz = 20.0
        self._last_emit = 0.0
        self._last_status = 0.0
        self.received_captures = 0
        self.displayed_captures = 0
        self.skipped_display_captures = 0
        self.missing_capture_sequences = 0
        self._last_received_sequence: int | None = None
        self._last_displayed_sequence: int | None = None

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.port))
            sock.settimeout(0.25)
            if self.esp_host:
                sock.sendto(b"pslab-gui", (self.esp_host, self.port))
            self.status.emit(f"Wi-Fi receiver listening on UDP {self.port}")

            while not self._stop:
                try:
                    packet, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                if not self.esp_host:
                    sock.sendto(b"pslab-gui", addr)

                for offset in range(0, len(packet), FRAME_LEN):
                    frame = packet[offset : offset + FRAME_LEN]
                    if len(frame) == FRAME_LEN:
                        self._handle_frame(frame)
        except Exception as exc:  # pragma: no cover - UI path
            self.failed.emit(str(exc))
        finally:
            sock.close()

    def _handle_frame(self, frame: bytes) -> None:
        if frame[0] != OUTER_MAGIC or frame[1] != OUTER_TYPE_DATA:
            return
        payload_len = struct.unpack_from("<H", frame, 6)[0]
        if payload_len > OUTER_PAYLOAD_LEN or frame[8] != checksum8(frame[:8]):
            return
        payload = frame[OUTER_HEADER_LEN : OUTER_HEADER_LEN + payload_len]
        if len(payload) < PSLAB_HEADER_LEN or payload[:4] != PSLAB_MAGIC:
            return

        instrument = payload[5]
        subtype = payload[6]
        capture_seq = struct.unpack_from("<I", payload, 8)[0]
        chunk_index = struct.unpack_from("<H", payload, 12)[0]
        chunk_count = struct.unpack_from("<H", payload, 14)[0]
        data_len = struct.unpack_from("<H", payload, 16)[0]
        data = payload[PSLAB_HEADER_LEN : PSLAB_HEADER_LEN + data_len]

        if subtype == PSLAB_SUBTYPE_META and len(data) >= 24:
            rate, samples, channels, source, trigger, data_format = struct.unpack_from("<IIIIII", data, 0)
            self._meta = WifiMeta(
                instrument=instrument,
                capture_sequence=capture_seq,
                sample_rate=rate,
                sample_count=samples,
                channel_count=channels,
                source=source,
                trigger_mode=trigger,
                data_format=data_format,
            )
            self._chunks = {}
            self._chunk_count = 0
            return

        if subtype != PSLAB_SUBTYPE_DATA or self._meta is None:
            return
        if capture_seq != self._meta.capture_sequence:
            return

        self._chunks[chunk_index] = bytes(data)
        self._chunk_count = chunk_count
        if chunk_count and len(self._chunks) >= chunk_count:
            try:
                payload = b"".join(self._chunks[index] for index in range(chunk_count))
            except KeyError:
                self._chunks = {}
                return
            capture = WifiCapture(meta=self._meta, payload=payload)
            self.received_captures += 1
            if self._last_received_sequence is not None and capture.meta.capture_sequence > self._last_received_sequence + 1:
                self.missing_capture_sequences += capture.meta.capture_sequence - self._last_received_sequence - 1
            self._last_received_sequence = capture.meta.capture_sequence
            now = time.monotonic()
            if now - self._last_status >= 1.0:
                self.status.emit(
                    "Wi-Fi received="
                    f"{self.received_captures} displayed={self.displayed_captures} "
                    f"display_skip={self.skipped_display_captures} "
                    f"seq_gap={self.missing_capture_sequences}"
                )
                self._last_status = now
            min_interval = 1.0 / self.max_display_hz
            if now - self._last_emit < min_interval:
                self.skipped_display_captures += 1
                self._chunks = {}
                return
            self._last_emit = now
            self.displayed_captures += 1
            if self._last_displayed_sequence is not None and self._meta.capture_sequence > self._last_displayed_sequence + 1:
                self.skipped_display_captures += self._meta.capture_sequence - self._last_displayed_sequence - 1
            self._last_displayed_sequence = self._meta.capture_sequence
            try:
                if self._meta.instrument == 1:
                    logic = self._wifi_to_logic(capture)
                    self.capture_ready.emit(logic, self._meta.capture_sequence, "LA")
                elif self._meta.instrument == 2:
                    scope = self._wifi_to_scope(capture)
                    self.capture_ready.emit(scope, self._meta.capture_sequence, "DSO")
            except Exception:
                self.failed.emit(log_exception("Wi-Fi capture decode failed"))
            self._chunks = {}

    def _wifi_to_logic(self, capture: WifiCapture) -> LogicCapture:
        meta = capture.meta
        words = np.frombuffer(capture.payload, dtype="<u4").copy()
        bits_per_word = 32 - (32 % max(meta.channel_count, 1))
        expected_words = int(np.ceil((meta.sample_count * meta.channel_count) / bits_per_word))
        if words.size < expected_words:
            raise ValueError(f"short LA payload: {words.size} words, expected at least {expected_words}")
        divider = max(1, round(PicoLogicAnalyzer.DEFAULT_SYS_CLOCK_HZ / max(meta.sample_rate, 1)))
        states = PicoLogicAnalyzer.decode_words(words, meta.channel_count, meta.sample_count)
        return LogicCapture(
            pin_base=meta.source,
            pin_count=meta.channel_count,
            samples=meta.sample_count,
            divider=divider,
            trigger_pin=meta.source,
            trigger_level=True,
            trigger_mode="edge" if meta.trigger_mode == 1 else "level",
            words=words,
            states=states,
            sample_rate=float(meta.sample_rate),
        )

    def _wifi_to_scope(self, capture: WifiCapture) -> ScopeCapture:
        meta = capture.meta
        raw = np.frombuffer(capture.payload, dtype="<u2").copy()
        return ScopeCapture(
            channel=meta.source,
            gpio=26 + meta.source,
            sample_rate=float(meta.sample_rate),
            samples=raw.size,
            trigger_level=0,
            trigger_mode="OFF",
            trigger_slope="RISE",
            raw=raw,
        )


class StreamStatusPoller(QtCore.QThread):
    status_ready = QtCore.pyqtSignal(str, str)

    def __init__(self, port: str | None):
        super().__init__()
        self.port = port
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            try:
                device = PicoLogicAnalyzer(port=self.port, timeout=0.4)
                cstream = device.query("LA:CSTREAM:STAT?")
                wifi = device.query("COMM:WIFI:STAT?")
                device.disconnect()
                self.status_ready.emit(cstream, wifi)
            except Exception as exc:
                self.status_ready.emit(f"poll-error:{exc}", "")

            for _ in range(10):
                if self._stop:
                    break
                self.msleep(100)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("PSLab Pico Waveform Workbench")
        self.resize(1400, 850)
        self.setMinimumSize(900, 560)
        self.capture_worker: CaptureWorker | None = None
        self.wifi_worker: WifiReceiver | None = None
        self.status_poller: StreamStatusPoller | None = None
        self.wifi_stream_instrument: str | None = None
        self.wifi_stream_command_prefix: str | None = None
        self.last_capture: LogicCapture | ScopeCapture | None = None
        self.channel_checks: list[QtWidgets.QCheckBox] = []
        self.pending_capture: tuple[object, int, str] | None = None
        self.logic_history: list[tuple[int | None, LogicCapture, float]] = []
        self.scope_history: list[tuple[int | None, ScopeCapture, float]] = []
        self.logic_next_offset_us = 0.0
        self.scope_next_offset_us = 0.0
        self.gui_sequence_gaps = 0
        self._last_gui_sequence: int | None = None
        self._last_channel_signature: tuple[int, int] | None = None
        self._build_ui()
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setInterval(33)
        self.render_timer.timeout.connect(self.render_pending_capture)
        self.render_timer.start()
        self.refresh_ports()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_F11:
            self.toggle_fullscreen()
            return
        if event.key() == QtCore.Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
        super().keyPressEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(360)
        controls.setMinimumWidth(320)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        root.addWidget(controls)

        self.view = WaveformView()
        root.addWidget(self.view, 1)

        self._build_connection_controls(controls_layout)
        self._build_transport_controls(controls_layout)
        self._build_instrument_controls(controls_layout)
        self._build_view_controls(controls_layout)
        controls_layout.addStretch(1)

        self.status_label = QtWidgets.QLabel("Ready")
        self.stream_status_label = QtWidgets.QLabel("Stream: idle")
        self.stream_status_label.setMinimumWidth(520)
        self.statusBar().addPermanentWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.stream_status_label, 2)

    def _build_connection_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        box = QtWidgets.QGroupBox("Connection")
        form = QtWidgets.QFormLayout(box)
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_ports)
        port_row = QtWidgets.QHBoxLayout()
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(refresh)
        form.addRow("USB CDC", port_row)
        identify = QtWidgets.QPushButton("Identify")
        identify.clicked.connect(self.identify)
        form.addRow(identify)
        layout.addWidget(box)

    def _build_transport_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        box = QtWidgets.QGroupBox("Transport")
        form = QtWidgets.QFormLayout(box)
        self.transport_combo = QtWidgets.QComboBox()
        self.transport_combo.addItems(["USB", "WIFI", "AUTO"])
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_transport)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.transport_combo, 1)
        row.addWidget(apply_button)
        form.addRow("Data path", row)
        wifi_status = QtWidgets.QPushButton("Wi-Fi Status")
        wifi_status.clicked.connect(self.query_wifi_status)
        form.addRow(wifi_status)
        self.udp_port = QtWidgets.QSpinBox()
        self.udp_port.setRange(1, 65535)
        self.udp_port.setValue(5005)
        self.esp_host = QtWidgets.QLineEdit()
        self.esp_host.setPlaceholderText("optional ESP IP")
        form.addRow("UDP port", self.udp_port)
        form.addRow("ESP IP", self.esp_host)
        self.wifi_button = QtWidgets.QPushButton("Start Wi-Fi Receiver")
        self.wifi_button.clicked.connect(self.toggle_wifi_receiver)
        form.addRow(self.wifi_button)
        layout.addWidget(box)

    def _build_instrument_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._logic_tab(), "Logic")
        self.tabs.addTab(self._scope_tab(), "Scope")
        layout.addWidget(self.tabs)

        actions = QtWidgets.QGroupBox("Capture")
        action_layout = QtWidgets.QGridLayout(actions)
        self.capture_button = QtWidgets.QPushButton("Capture")
        self.capture_button.clicked.connect(lambda: self.start_capture(False))
        self.repeat_button = QtWidgets.QPushButton("Repeat")
        self.repeat_button.setCheckable(True)
        self.repeat_button.clicked.connect(self.toggle_repeat)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_capture)
        self.repeat_delay = QtWidgets.QSpinBox()
        self.repeat_delay.setRange(0, 5000)
        self.repeat_delay.setValue(20)
        self.repeat_delay.setSuffix(" ms")
        action_layout.addWidget(self.capture_button, 0, 0)
        action_layout.addWidget(self.repeat_button, 0, 1)
        action_layout.addWidget(self.stop_button, 0, 2)
        action_layout.addWidget(QtWidgets.QLabel("Delay"), 1, 0)
        action_layout.addWidget(self.repeat_delay, 1, 1, 1, 2)
        layout.addWidget(actions)

    def _logic_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.la_pin_base = self._spin(0, 29, 16)
        self.la_pin_count = self._spin(1, 8, 2)
        self.la_samples = self._spin(1, 4096, 1024)
        self.la_divider = self._spin(1, 16_777_215, 150)
        self.la_trigger_pin = self._spin(0, 29, 16)
        self.la_trigger_level = QtWidgets.QComboBox()
        self.la_trigger_level.addItems(["1", "0"])
        self.la_trigger_mode = QtWidgets.QComboBox()
        self.la_trigger_mode.addItems(["EDGE", "LEVEL"])
        self.sys_clock = self._spin(1, 500_000_000, 150_000_000)
        self.test_pin = self._spin(0, 29, 15)
        self.test_freq = self._spin(1, 20_000_000, 1000)
        for label, control in [
            ("Pin base", self.la_pin_base),
            ("Pin count", self.la_pin_count),
            ("Samples", self.la_samples),
            ("Divider", self.la_divider),
            ("Trigger pin", self.la_trigger_pin),
            ("Trigger level", self.la_trigger_level),
            ("Trigger mode", self.la_trigger_mode),
            ("System clock", self.sys_clock),
        ]:
            form.addRow(label, control)
        test_row = QtWidgets.QHBoxLayout()
        start_test = QtWidgets.QPushButton("Start Test")
        stop_test = QtWidgets.QPushButton("Stop")
        start_test.clicked.connect(self.start_test_signal)
        stop_test.clicked.connect(self.stop_test_signal)
        test_row.addWidget(self.test_pin)
        test_row.addWidget(self.test_freq)
        test_row.addWidget(start_test)
        test_row.addWidget(stop_test)
        form.addRow("Test pin/Hz", test_row)
        return widget

    def _scope_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.dso_channel = self._spin(0, 3, 0)
        self.dso_samples = self._spin(1, 4096, 1024)
        self.dso_rate = self._spin(1, 500_000, 100_000)
        self.dso_trigger_level = self._spin(0, 4095, 2048)
        self.dso_trigger_mode = QtWidgets.QComboBox()
        self.dso_trigger_mode.addItems(["OFF", "LEVEL", "EDGE"])
        self.dso_trigger_slope = QtWidgets.QComboBox()
        self.dso_trigger_slope.addItems(["RISE", "FALL"])
        for label, control in [
            ("Channel", self.dso_channel),
            ("Samples", self.dso_samples),
            ("Rate", self.dso_rate),
            ("Trigger level", self.dso_trigger_level),
            ("Trigger mode", self.dso_trigger_mode),
            ("Trigger slope", self.dso_trigger_slope),
        ]:
            form.addRow(label, control)
        return widget

    def _build_view_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        box = QtWidgets.QGroupBox("View")
        view_layout = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        for text, slot in [
            ("Zoom +", self.view.zoom_in),
            ("Zoom -", self.view.zoom_out),
            ("Fit", self.view.fit_time),
        ]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        full_button = QtWidgets.QPushButton("Full Screen")
        full_button.clicked.connect(self.toggle_fullscreen)
        row.addWidget(full_button)
        view_layout.addLayout(row)
        self.autoscroll = QtWidgets.QCheckBox("Auto-scroll to newest capture")
        self.autoscroll.setChecked(True)
        self.autoscroll.toggled.connect(self.view.set_autoscroll)
        self.view.set_autoscroll(True)
        view_layout.addWidget(self.autoscroll)
        self.append_timeline = QtWidgets.QCheckBox("Rolling timeline")
        self.append_timeline.setChecked(False)
        self.append_timeline.setToolTip(
            "Append displayed captures to the right. Leave off for the most faithful latest-buffer view."
        )
        view_layout.addWidget(self.append_timeline)
        self.preserve_gaps = QtWidgets.QCheckBox("Show skipped capture gaps")
        self.preserve_gaps.setChecked(False)
        self.preserve_gaps.toggled.connect(self.view.set_preserve_sequence_gaps)
        view_layout.addWidget(self.preserve_gaps)
        self.history_limit = QtWidgets.QSpinBox()
        self.history_limit.setRange(1, 250)
        self.history_limit.setValue(20)
        self.history_limit.setSuffix(" captures")
        view_layout.addWidget(self.history_limit)
        self.display_rate = QtWidgets.QSpinBox()
        self.display_rate.setRange(1, 60)
        self.display_rate.setValue(15)
        self.display_rate.setSuffix(" FPS")
        self.display_rate.valueChanged.connect(self.update_wifi_display_rate)
        view_layout.addWidget(self.display_rate)
        self.use_multicore_cstream = QtWidgets.QCheckBox("Use multicore LA:CSTREAM")
        self.use_multicore_cstream.setChecked(False)
        view_layout.addWidget(self.use_multicore_cstream)
        clear_button = QtWidgets.QPushButton("Clear Timeline")
        clear_button.clicked.connect(self.clear_timeline)
        view_layout.addWidget(clear_button)
        self.channel_box = QtWidgets.QGroupBox("Channels")
        self.channel_layout = QtWidgets.QVBoxLayout(self.channel_box)
        view_layout.addWidget(self.channel_box)
        layout.addWidget(box)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        self.port_combo.addItems(available_ports())
        if current:
            self.port_combo.setCurrentText(current)

    def selected_port(self) -> str | None:
        text = self.port_combo.currentText().strip()
        return text or None

    def logic_config(self) -> dict:
        return {
            "pin_base": self.la_pin_base.value(),
            "pin_count": self.la_pin_count.value(),
            "samples": self.la_samples.value(),
            "divider": self.la_divider.value(),
            "trigger_pin": self.la_trigger_pin.value(),
            "trigger_level": self.la_trigger_level.currentText() == "1",
            "trigger_mode": self.la_trigger_mode.currentText().lower(),
            "sys_clock_hz": self.sys_clock.value(),
            "repeat_delay_ms": self.repeat_delay.value(),
        }

    def scope_config(self) -> dict:
        return {
            "dso_channel": self.dso_channel.value(),
            "dso_samples": self.dso_samples.value(),
            "dso_rate": self.dso_rate.value(),
            "dso_trigger_level": self.dso_trigger_level.value(),
            "dso_trigger_mode": self.dso_trigger_mode.currentText(),
            "dso_trigger_slope": self.dso_trigger_slope.currentText(),
            "repeat_delay_ms": self.repeat_delay.value(),
        }

    def start_capture(self, repeat: bool) -> None:
        self.stop_capture()
        if not repeat:
            self.clear_timeline()
        instrument = "LA" if self.tabs.currentIndex() == 0 else "DSO"
        config = self.logic_config() if instrument == "LA" else self.scope_config()
        self.capture_worker = CaptureWorker(instrument, self.selected_port(), config, repeat)
        self.capture_worker.capture_ready.connect(self.on_capture_ready)
        self.capture_worker.status.connect(self.set_status)
        self.capture_worker.failed.connect(self.show_error)
        self.capture_worker.finished_cleanly.connect(self.on_capture_finished)
        self.capture_worker.start()
        self.set_status(f"{instrument} capture started")

    def toggle_repeat(self, checked: bool) -> None:
        if checked:
            if self.transport_combo.currentText() in ("WIFI", "AUTO"):
                self.start_wireless_stream()
            else:
                self.start_capture(True)
            self.repeat_button.setText("Repeating")
        else:
            self.stop_capture()

    def stop_capture(self) -> None:
        if self.capture_worker is not None:
            self.capture_worker.stop()
            self.capture_worker.wait(200)
            self.capture_worker = None
        self.stop_status_poller()
        if self.wifi_stream_instrument is not None:
            try:
                device = PicoLogicAnalyzer(port=self.selected_port(), timeout=1.0)
                if self.wifi_stream_command_prefix == "LA:CSTREAM":
                    device.command("LA:CSTREAM:STOP")
                elif self.wifi_stream_instrument == "LA":
                    device.command("LA:STREAM:STOP")
                else:
                    device.command("DSO:STREAM:STOP")
                device.disconnect()
            except Exception as exc:
                self.set_status(f"Stream stop warning: {exc}")
            self.wifi_stream_instrument = None
            self.wifi_stream_command_prefix = None
            self.stream_status_label.setText("Stream: stopped")
        self.repeat_button.setChecked(False)
        self.repeat_button.setText("Repeat")

    def start_wireless_stream(self) -> None:
        if self.wifi_worker is None:
            self.toggle_wifi_receiver()

        instrument = "LA" if self.tabs.currentIndex() == 0 else "DSO"
        try:
            self.clear_timeline()
            if instrument == "LA":
                config = self.logic_config()
                device = PicoLogicAnalyzer(
                    port=self.selected_port(),
                    timeout=2.0,
                    sys_clock_hz=config["sys_clock_hz"],
                )
                device.configure(
                    pin_base=config["pin_base"],
                    pin_count=config["pin_count"],
                    samples=config["samples"],
                    divider=config["divider"],
                    trigger_pin=config["trigger_pin"],
                    trigger_level=config["trigger_level"],
                    trigger_mode=config["trigger_mode"],
                )
                if self.use_multicore_cstream.isChecked():
                    if self.append_timeline.isChecked():
                        self.history_limit.setValue(min(self.history_limit.value(), 8))
                        self.display_rate.setValue(min(self.display_rate.value(), 10))
                        self.update_wifi_display_rate(self.display_rate.value())
                    device.command("LA:CSTREAM:START")
                    self.wifi_stream_command_prefix = "LA:CSTREAM"
                else:
                    device.stream_start()
                    self.wifi_stream_command_prefix = "LA:STREAM"
            else:
                config = self.scope_config()
                device = PicoOscilloscope(port=self.selected_port(), timeout=2.0)
                device.configure(
                    channel=config["dso_channel"],
                    sample_rate=config["dso_rate"],
                    samples=config["dso_samples"],
                    trigger_level=config["dso_trigger_level"],
                    trigger_mode=config["dso_trigger_mode"],
                    trigger_slope=config["dso_trigger_slope"],
                )
                device.command("DSO:STREAM:START")
                self.wifi_stream_command_prefix = "DSO:STREAM"
            device.disconnect()
            self.wifi_stream_instrument = instrument
            if self.wifi_stream_command_prefix == "LA:CSTREAM":
                self.start_status_poller()
            self.set_status(f"{instrument} Wi-Fi stream started")
        except Exception as exc:
            self.repeat_button.setChecked(False)
            self.repeat_button.setText("Repeat")
            self.show_error(str(exc))

    def on_capture_finished(self) -> None:
        self.repeat_button.setText("Repeat" if not self.repeat_button.isChecked() else "Repeating")

    def on_capture_ready(self, capture, sequence: int, instrument: str) -> None:
        self.pending_capture = (capture, sequence, instrument)

    def start_status_poller(self) -> None:
        self.stop_status_poller()
        self.gui_sequence_gaps = 0
        self._last_gui_sequence = None
        self.status_poller = StreamStatusPoller(self.selected_port())
        self.status_poller.status_ready.connect(self.on_stream_status_ready)
        self.status_poller.start()

    def stop_status_poller(self) -> None:
        if self.status_poller is not None:
            self.status_poller.stop()
            self.status_poller.wait(700)
            self.status_poller = None

    def on_stream_status_ready(self, cstream: str, wifi: str) -> None:
        self.stream_status_label.setText(
            f"CSTREAM {self._format_cstream_status(cstream)} | WIFI {wifi or '-'} | GUI gaps={self.gui_sequence_gaps}"
        )

    @staticmethod
    def _format_cstream_status(status: str) -> str:
        parts = [part.strip() for part in status.split(",")]
        if len(parts) != 6:
            return status
        requested, running, sequence, sent, overruns, errors = parts
        return (
            f"req={requested} run={running} seq={sequence} "
            f"sent={sent} over={overruns} err={errors}"
        )

    def render_pending_capture(self) -> None:
        try:
            self._render_pending_capture()
        except Exception:
            self.pending_capture = None
            self.show_error(log_exception("GUI render failed"))

    def _render_pending_capture(self) -> None:
        if self.pending_capture is None:
            return

        capture, sequence, instrument = self.pending_capture
        self.pending_capture = None
        self.last_capture = capture
        if self._last_gui_sequence is not None and sequence > self._last_gui_sequence + 1:
            self.gui_sequence_gaps += sequence - self._last_gui_sequence - 1
        self._last_gui_sequence = sequence
        if instrument == "LA":
            signature = (capture.pin_base, capture.pin_count)
            if signature != self._last_channel_signature:
                self.update_channel_controls(capture)
                self._last_channel_signature = signature

            if self.append_timeline.isChecked():
                self._append_logic_history(sequence, capture)
                self.logic_history = self.logic_history[-self.history_limit.value():]
                self.view.plot_logic_history(self.logic_history)
            else:
                self.logic_history = [(sequence, capture, 0.0)]
                self.logic_next_offset_us = capture.samples * capture.sample_interval_us
                self.view.plot_logic(capture, sequence)
        else:
            self.update_channel_controls(None)
            self._last_channel_signature = None
            if self.append_timeline.isChecked():
                self._append_scope_history(sequence, capture)
                self.scope_history = self.scope_history[-self.history_limit.value():]
                self.view.plot_scope_history(self.scope_history)
            else:
                self.scope_history = [(sequence, capture, 0.0)]
                self.scope_next_offset_us = capture.samples * capture.sample_interval_us
                self.view.plot_scope(capture, sequence)
        self.set_status(f"{instrument} capture {sequence} plotted")

    def _append_logic_history(self, sequence: int | None, capture: LogicCapture) -> None:
        if self.logic_history and self.preserve_gaps.isChecked():
            previous_sequence, previous_capture, _ = self.logic_history[-1]
            if sequence is not None and previous_sequence is not None and sequence > previous_sequence + 1:
                previous_duration = previous_capture.samples * previous_capture.sample_interval_us
                self.logic_next_offset_us += previous_duration * min(sequence - previous_sequence - 1, 20)

        self.logic_history.append((sequence, capture, self.logic_next_offset_us))
        self.logic_next_offset_us += capture.samples * capture.sample_interval_us

    def _append_scope_history(self, sequence: int | None, capture: ScopeCapture) -> None:
        if self.scope_history and self.preserve_gaps.isChecked():
            previous_sequence, previous_capture, _ = self.scope_history[-1]
            if sequence is not None and previous_sequence is not None and sequence > previous_sequence + 1:
                previous_duration = previous_capture.samples * previous_capture.sample_interval_us
                self.scope_next_offset_us += previous_duration * min(sequence - previous_sequence - 1, 20)

        self.scope_history.append((sequence, capture, self.scope_next_offset_us))
        self.scope_next_offset_us += capture.samples * capture.sample_interval_us

    def clear_timeline(self) -> None:
        self.logic_history.clear()
        self.scope_history.clear()
        self.logic_next_offset_us = 0.0
        self.scope_next_offset_us = 0.0
        self.gui_sequence_gaps = 0
        self._last_gui_sequence = None
        self.pending_capture = None
        self.view.x_scale = self.view.base_x_scale
        self.view.scene().clear()

    def update_channel_controls(self, capture: LogicCapture | None) -> None:
        while self.channel_layout.count():
            item = self.channel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.channel_checks = []
        if capture is None:
            return
        for i in range(capture.pin_count):
            name = f"GPIO{capture.pin_base + i}"
            check = QtWidgets.QCheckBox(name)
            check.setChecked(self.view.channel_visible.get(name, True))
            check.toggled.connect(lambda checked, n=name: self.on_channel_toggled(n, checked))
            self.channel_layout.addWidget(check)
            self.channel_checks.append(check)

    def on_channel_toggled(self, name: str, checked: bool) -> None:
        self.view.set_channel_visible(name, checked)
        if isinstance(self.last_capture, LogicCapture):
            if self.append_timeline.isChecked() and self.logic_history:
                self.view.plot_logic_history(self.logic_history)
            else:
                self.view.plot_logic(self.last_capture)

    def identify(self) -> None:
        try:
            device = PicoLogicAnalyzer(port=self.selected_port(), timeout=1.0)
            identity = device.identify()
            device.disconnect()
            self.set_status(identity)
        except Exception as exc:
            self.show_error(str(exc))

    def apply_transport(self) -> None:
        try:
            device = PicoLogicAnalyzer(port=self.selected_port(), timeout=1.0)
            device.command(f"COMM:TRAN {self.transport_combo.currentText()}")
            mode = device.query("COMM:TRAN?")
            device.disconnect()
            self.set_status(f"Transport set to {mode}")
        except Exception as exc:
            self.show_error(str(exc))

    def query_wifi_status(self) -> None:
        try:
            device = PicoLogicAnalyzer(port=self.selected_port(), timeout=1.0)
            status = device.query("COMM:WIFI:STAT?")
            device.disconnect()
            self.set_status(f"Wi-Fi status {status}")
        except Exception as exc:
            self.show_error(str(exc))

    def toggle_wifi_receiver(self) -> None:
        if self.wifi_worker is not None:
            self.wifi_worker.stop()
            self.wifi_worker.wait(500)
            self.wifi_worker = None
            self.wifi_button.setText("Start Wi-Fi Receiver")
            self.set_status("Wi-Fi receiver stopped")
            return

        esp = self.esp_host.text().strip() or None
        self.wifi_worker = WifiReceiver(port=self.udp_port.value(), esp_host=esp)
        self.wifi_worker.max_display_hz = float(self.display_rate.value())
        self.wifi_worker.capture_ready.connect(self.on_capture_ready)
        self.wifi_worker.status.connect(self.set_status)
        self.wifi_worker.failed.connect(self.show_error)
        self.wifi_worker.start()
        self.wifi_button.setText("Stop Wi-Fi Receiver")

    def update_wifi_display_rate(self, value: int) -> None:
        if self.wifi_worker is not None:
            self.wifi_worker.max_display_hz = float(value)

    def start_test_signal(self) -> None:
        try:
            device = PicoLogicAnalyzer(port=self.selected_port(), timeout=1.0)
            device.start_test_square(self.test_pin.value(), self.test_freq.value())
            device.disconnect()
            self.set_status("Test square started")
        except Exception as exc:
            self.show_error(str(exc))

    def stop_test_signal(self) -> None:
        try:
            device = PicoLogicAnalyzer(port=self.selected_port(), timeout=1.0)
            device.stop_test_square()
            device.disconnect()
            self.set_status("Test square stopped")
        except Exception as exc:
            self.show_error(str(exc))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def show_error(self, text: str) -> None:
        self.set_status(text)
        QtWidgets.QMessageBox.warning(self, "PSLab Pico", text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_capture()
        self.stop_status_poller()
        if self.wifi_worker is not None:
            self.wifi_worker.stop()
            self.wifi_worker.wait(500)
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("PSLab Pico Waveform Workbench")
    window = MainWindow()
    if args.port:
        window.port_combo.setCurrentText(args.port)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
