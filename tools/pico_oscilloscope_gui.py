#!/usr/bin/env python3
"""Development GUI for the PSLab Pico oscilloscope."""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pslab-pico")

try:
    from PyQt5 import QtCore, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from serial.tools import list_ports
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing import guard
    raise SystemExit(
        "Missing GUI dependency. Install PyQt5 and matplotlib, then run again.\n"
        "Example: python3 -m pip install PyQt5 matplotlib"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pslab.instrument.pico_oscilloscope import PicoOscilloscope, ScopeCapture


SCROLL_STEPS = 10000


@dataclass
class ScopeSettings:
    port: str | None
    channel: int
    sample_rate: int
    samples: int
    trigger_level: int
    trigger_mode: str
    trigger_slope: str


def measure_scope(raw: np.ndarray, sample_rate: float) -> dict[str, float | int | None]:
    if raw.size == 0:
        return {"min": None, "max": None, "mean": None, "pp": None, "frequency": None}

    volts = raw.astype(np.float64) * (3.3 / 4095.0)
    centered = raw > float(np.mean(raw))
    edges = np.flatnonzero(centered[1:] != centered[:-1]) + 1
    rising = edges[centered[edges]]
    frequency = None
    if rising.size >= 2:
        mean_period = float(np.mean(np.diff(rising) / sample_rate))
        if mean_period > 0.0:
            frequency = 1.0 / mean_period

    return {
        "min": float(np.min(volts)),
        "max": float(np.max(volts)),
        "mean": float(np.mean(volts)),
        "pp": float(np.max(volts) - np.min(volts)),
        "frequency": frequency,
        "edges": int(edges.size),
    }


class ScopeCaptureWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, float)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings: ScopeSettings):
        super().__init__()
        self.settings = settings

    @QtCore.pyqtSlot()
    def run(self) -> None:
        start = time.perf_counter()
        try:
            scope = PicoOscilloscope(port=self.settings.port, timeout=2.0)
            capture = scope.capture(
                channel=self.settings.channel,
                sample_rate=self.settings.sample_rate,
                samples=self.settings.samples,
                trigger_level=self.settings.trigger_level,
                trigger_mode=self.settings.trigger_mode,
                trigger_slope=self.settings.trigger_slope,
            )
            scope.disconnect()
        except Exception as exc:  # pragma: no cover - GUI error path
            self.failed.emit(str(exc))
            return

        self.finished.emit(capture, time.perf_counter() - start)


class ScopeStreamWorker(QtCore.QObject):
    frame = QtCore.pyqtSignal(int, object, float)
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, settings: ScopeSettings):
        super().__init__()
        self.settings = settings
        self._stop_requested = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        scope = PicoOscilloscope(port=self.settings.port, timeout=2.0)
        try:
            scope.configure(
                channel=self.settings.channel,
                sample_rate=self.settings.sample_rate,
                samples=self.settings.samples,
                trigger_level=self.settings.trigger_level,
                trigger_mode=self.settings.trigger_mode,
                trigger_slope=self.settings.trigger_slope,
            )
            scope.stream_start()
            while not self._stop_requested:
                start = time.perf_counter()
                sequence, capture = scope.read_stream_frame()
                self.frame.emit(sequence, capture, time.perf_counter() - start)
        except Exception as exc:  # pragma: no cover - GUI error path
            if not self._stop_requested:
                self.failed.emit(str(exc))
        finally:
            try:
                scope.stream_stop()
            except Exception:
                pass
            scope.disconnect()
            self.stopped.emit()

    def request_stop(self) -> None:
        self._stop_requested = True


class ScopeCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def plot_capture(
        self,
        capture: ScopeCapture | None,
        *,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        if capture is None:
            self.plot_series(np.array([]), np.array([]), "No capture")
            return
        self.plot_series(
            capture.time_us,
            capture.volts,
            (
                f"ADC{capture.channel} / GPIO{capture.gpio}, "
                f"{capture.sample_rate / 1000:.1f} kS/s, {capture.samples} samples"
            ),
            xlim=xlim,
        )

    def plot_series(
        self,
        time_us: np.ndarray,
        volts: np.ndarray,
        title: str,
        *,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        self.axes.clear()
        if time_us.size and volts.size:
            self.axes.plot(time_us, volts, linewidth=1.1)
            self.axes.set_ylabel("Voltage (V)")
            self.axes.set_ylim(-0.1, 3.4)
            if xlim is not None and xlim[1] > xlim[0]:
                self.axes.set_xlim(*xlim)
        self.axes.set_xlabel("Time (us)")
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.25)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSLab Pico Oscilloscope")
        self.resize(1500, 920)

        self.capture: ScopeCapture | None = None
        self.worker_thread: QtCore.QThread | None = None
        self.worker: ScopeCaptureWorker | None = None
        self.stream_thread: QtCore.QThread | None = None
        self.stream_worker: ScopeStreamWorker | None = None
        self.live_enabled = False
        self.live_time_us = np.array([], dtype=float)
        self.live_volts = np.array([], dtype=float)
        self.live_sample_rate = 1.0
        self.scroll_data_start_us = 0.0
        self.scroll_data_end_us = 0.0
        self.scroll_window_us = 0.0
        self.updating_scrollbar = False

        QtWidgets.QShortcut("F11", self, activated=self.toggle_fullscreen)
        self._apply_style()
        self._build_ui()
        self.refresh_ports()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(self._top_bar())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        controls = self._controls()
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setMinimumWidth(340)
        controls_scroll.setMaximumWidth(520)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        plot_area = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        self.plot = ScopeCanvas()
        self.toolbar = NavigationToolbar(self.plot, self)
        self.plot_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.plot_scrollbar.setRange(0, 0)
        self.plot_scrollbar.valueChanged.connect(self.plot_scrollbar_changed)
        self.plot_scrollbar.sliderPressed.connect(self.plot_scrollbar_pressed)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.plot, stretch=1)
        plot_layout.addWidget(self.plot_scrollbar)
        splitter.addWidget(plot_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1110])

        self.status = self.statusBar()
        self._show_status("Ready")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { font-size: 10pt; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #c9ced6;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton { min-height: 28px; padding: 4px 10px; }
            QPushButton[primary="true"] { font-weight: 700; min-height: 34px; }
            QLabel#BrandLabel { font-size: 15pt; font-weight: 800; }
            QLabel#StatusPill {
                border: 1px solid #c9ced6;
                border-radius: 12px;
                padding: 4px 10px;
                background: #f5f7fa;
            }
            QPlainTextEdit { font-family: monospace; }
            """
        )

    def _top_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("PSLab Pico")
        title.setObjectName("BrandLabel")
        subtitle = QtWidgets.QLabel("Oscilloscope")
        subtitle.setStyleSheet("color: #586171;")
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)

        self.capture_button = QtWidgets.QPushButton("Capture")
        self.live_button = QtWidgets.QPushButton("Start Live")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.save_button = QtWidgets.QPushButton("Save CSV")
        self.fullscreen_button = QtWidgets.QPushButton("Full Screen")
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setObjectName("StatusPill")

        self.capture_button.setProperty("primary", True)
        self.live_button.setProperty("primary", True)
        self.capture_button.clicked.connect(self.start_capture)
        self.live_button.clicked.connect(self.toggle_live)
        self.clear_button.clicked.connect(self.clear_display)
        self.save_button.clicked.connect(self.save_csv)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        layout.addWidget(self.capture_button)
        layout.addWidget(self.live_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.save_button)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.fullscreen_button)
        return bar

    def _controls(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        setup = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(setup)
        setup_layout.setContentsMargins(8, 8, 8, 8)
        setup_layout.addWidget(self._connection_group())
        setup_layout.addWidget(self._capture_group())
        setup_layout.addWidget(self._display_group())
        setup_layout.addStretch(1)

        measure = QtWidgets.QWidget()
        measure_layout = QtWidgets.QVBoxLayout(measure)
        measure_layout.setContentsMargins(8, 8, 8, 8)
        measure_layout.addWidget(self._measurements_group())

        tabs.addTab(setup, "Setup")
        tabs.addTab(measure, "Measure")
        return tabs

    def _connection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Connection")
        layout = QtWidgets.QGridLayout(group)
        self.port_combo = QtWidgets.QComboBox()
        refresh = QtWidgets.QPushButton("Refresh")
        identify = QtWidgets.QPushButton("Connect")
        self.idn_label = QtWidgets.QLabel("Not connected")
        self.idn_label.setWordWrap(True)
        refresh.clicked.connect(self.refresh_ports)
        identify.clicked.connect(self.identify_device)
        layout.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        layout.addWidget(self.port_combo, 0, 1, 1, 2)
        layout.addWidget(refresh, 1, 1)
        layout.addWidget(identify, 1, 2)
        layout.addWidget(self.idn_label, 2, 0, 1, 3)
        return group

    def _capture_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Capture")
        layout = QtWidgets.QGridLayout(group)
        self.channel = self._spin(0, 3, 0)
        self.sample_rate = self._spin(1, 500_000, 100_000, 10_000)
        self.samples = self._spin(1, 4096, 1024, 256)
        self.trigger_level = self._spin(0, 4095, 2048, 128)
        self.trigger_mode = QtWidgets.QComboBox()
        self.trigger_mode.addItems(["OFF", "LEVEL", "EDGE"])
        self.trigger_slope = QtWidgets.QComboBox()
        self.trigger_slope.addItems(["RISE", "FALL"])
        self.window_label = QtWidgets.QLabel()
        self.gpio_label = QtWidgets.QLabel("GPIO26")
        self.channel.valueChanged.connect(self._update_gpio_label)
        self.sample_rate.valueChanged.connect(self._update_window_label)
        self.samples.valueChanged.connect(self._update_window_label)
        rows = [
            ("ADC channel", self.channel),
            ("Input GPIO", self.gpio_label),
            ("Sample rate", self.sample_rate),
            ("Samples", self.samples),
            ("Window", self.window_label),
            ("Trigger level", self.trigger_level),
            ("Trigger mode", self.trigger_mode),
            ("Trigger slope", self.trigger_slope),
        ]
        for row, (label, widget) in enumerate(rows):
            layout.addWidget(QtWidgets.QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)
        self._update_gpio_label()
        self._update_window_label()
        return group

    def _display_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Display")
        layout = QtWidgets.QGridLayout(group)
        self.auto_scroll = QtWidgets.QCheckBox("Auto-scroll live")
        self.auto_scroll.setChecked(True)
        self.live_window_ms = self._spin(1, 60000, 250, 50)
        self.retain_seconds = self._spin(1, 600, 10, 1)
        self.auto_scroll.stateChanged.connect(self._display_changed)
        self.live_window_ms.valueChanged.connect(self._display_changed)
        self.retain_seconds.valueChanged.connect(self._trim_live_history)
        layout.addWidget(self.auto_scroll, 0, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Visible window ms"), 1, 0)
        layout.addWidget(self.live_window_ms, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Retain seconds"), 2, 0)
        layout.addWidget(self.retain_seconds, 2, 1)
        return group

    def _measurements_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Measurements")
        layout = QtWidgets.QVBoxLayout(group)
        self.measurements = QtWidgets.QPlainTextEdit()
        self.measurements.setReadOnly(True)
        self.measurements.setMinimumHeight(220)
        layout.addWidget(self.measurements)
        return group

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, step: int = 1) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    def refresh_ports(self) -> None:
        selected = self.port_combo.currentData()
        self.port_combo.clear()
        self.port_combo.addItem("Auto detect", None)
        for port in list_ports.comports():
            label = port.device
            if port.product:
                label += f" - {port.product}"
            self.port_combo.addItem(label, port.device)
        if selected:
            index = self.port_combo.findData(selected)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

    def identify_device(self) -> None:
        try:
            scope = PicoOscilloscope(port=self.port_combo.currentData())
            identity = scope.identify()
            scope.disconnect()
        except Exception as exc:
            self.idn_label.setText(f"Connection failed: {exc}")
            self._show_status("Connection failed")
            return
        self.idn_label.setText(identity)
        self._show_status("Connected")

    def current_settings(self) -> ScopeSettings:
        return ScopeSettings(
            port=self.port_combo.currentData(),
            channel=self.channel.value(),
            sample_rate=self.sample_rate.value(),
            samples=self.samples.value(),
            trigger_level=self.trigger_level.value(),
            trigger_mode=self.trigger_mode.currentText(),
            trigger_slope=self.trigger_slope.currentText(),
        )

    def start_capture(self) -> None:
        if self.worker_thread is not None or self.stream_thread is not None:
            return
        self.capture_button.setEnabled(False)
        self._show_status("Capturing...")
        self.worker_thread = QtCore.QThread(self)
        self.worker = ScopeCaptureWorker(self.current_settings())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.capture_finished)
        self.worker.failed.connect(self.capture_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.worker_done)
        self.worker_thread.start()

    def capture_finished(self, capture: ScopeCapture, elapsed: float) -> None:
        self.capture = capture
        self.plot_capture_with_scroll(capture)
        self.update_measurements(capture, elapsed)
        self._show_status(f"Capture complete in {elapsed:.3f} s")

    def capture_failed(self, message: str) -> None:
        self._show_status(f"Capture failed: {message}")
        QtWidgets.QMessageBox.warning(self, "Capture failed", message)

    def worker_done(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.capture_button.setEnabled(True)

    def toggle_live(self) -> None:
        if self.live_enabled:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self) -> None:
        if self.stream_thread is not None:
            return
        self.clear_live_history()
        self.live_enabled = True
        self._set_live_button_text()
        self.capture_button.setEnabled(False)
        self._show_status("Starting stream...")
        self.stream_thread = QtCore.QThread(self)
        self.stream_worker = ScopeStreamWorker(self.current_settings())
        self.stream_worker.moveToThread(self.stream_thread)
        self.stream_thread.started.connect(self.stream_worker.run)
        self.stream_worker.frame.connect(self.stream_frame_received)
        self.stream_worker.failed.connect(self.stream_failed)
        self.stream_worker.stopped.connect(self.stream_thread.quit)
        self.stream_thread.finished.connect(self.stream_thread.deleteLater)
        self.stream_thread.finished.connect(self.stream_done)
        self.stream_thread.start()

    def stop_stream(self) -> None:
        self.live_enabled = False
        self._set_live_button_text()
        if self.stream_worker is not None:
            self.stream_worker.request_stop()
            self._show_status("Stopping stream...")

    def stream_frame_received(self, sequence: int, capture: ScopeCapture, elapsed: float) -> None:
        self.capture = capture
        self.append_live_capture(capture)
        self.plot_live()
        self.update_measurements(capture, elapsed)
        self._show_status(f"Streaming frame {sequence}")

    def stream_failed(self, message: str) -> None:
        self.live_enabled = False
        self._set_live_button_text()
        self._show_status(f"Stream failed: {message}")
        QtWidgets.QMessageBox.warning(self, "Stream failed", message)

    def stream_done(self) -> None:
        self.stream_worker = None
        self.stream_thread = None
        self.live_enabled = False
        self._set_live_button_text()
        self.capture_button.setEnabled(True)
        self._show_status("Live stopped")

    def append_live_capture(self, capture: ScopeCapture) -> None:
        sample_interval_us = capture.sample_interval_us
        if abs(self.live_sample_rate - capture.sample_rate) > 0.5:
            self.clear_live_history()
        self.live_sample_rate = capture.sample_rate
        start_us = float(self.live_time_us[-1] + sample_interval_us) if self.live_time_us.size else 0.0
        self.live_time_us = np.concatenate(
            [self.live_time_us, start_us + np.arange(capture.samples) * sample_interval_us]
        )
        self.live_volts = np.concatenate([self.live_volts, capture.volts])
        self._trim_live_history()

    def clear_live_history(self) -> None:
        self.live_time_us = np.array([], dtype=float)
        self.live_volts = np.array([], dtype=float)
        self.live_sample_rate = float(self.sample_rate.value())
        self.update_plot_scrollbar(0.0, 0.0, 0.0, follow=True)

    def clear_display(self) -> None:
        self.capture = None
        self.clear_live_history()
        self.measurements.clear()
        self.plot.plot_capture(None)
        self._show_status("Display cleared")

    def _trim_live_history(self, *_args) -> None:
        if self.live_time_us.size == 0:
            return
        retain_us = self.retain_seconds.value() * 1_000_000.0
        cutoff = self.live_time_us[-1] - retain_us
        first = int(np.searchsorted(self.live_time_us, cutoff, side="left"))
        if first > 0:
            self.live_time_us = self.live_time_us[first:]
            self.live_volts = self.live_volts[first:]

    def plot_live(self) -> None:
        if self.live_time_us.size == 0:
            self.plot.plot_capture(None)
            self.update_plot_scrollbar(0.0, 0.0, 0.0, follow=True)
            return
        xlim = self._live_xlim()
        self.update_plot_scrollbar(
            float(self.live_time_us[0]),
            float(self.live_time_us[-1]),
            self.live_window_ms.value() * 1000.0,
            follow=self.auto_scroll.isChecked(),
        )
        self.plot.plot_series(
            self.live_time_us,
            self.live_volts,
            f"Live ADC, {self.live_sample_rate / 1000:.1f} kS/s, {self.live_time_us.size} retained samples",
            xlim=xlim,
        )

    def plot_capture_with_scroll(self, capture: ScopeCapture) -> None:
        start = 0.0
        end = float(capture.time_us[-1]) if capture.time_us.size else 0.0
        window = min(end - start, self.live_window_ms.value() * 1000.0)
        if window <= 0.0:
            window = end - start
        self.update_plot_scrollbar(start, end, window, follow=False)
        self.plot.plot_capture(capture, xlim=self.scrollbar_xlim(start, end, window))

    def _live_xlim(self) -> tuple[float, float] | None:
        start = float(self.live_time_us[0])
        end = float(self.live_time_us[-1])
        window_us = self.live_window_ms.value() * 1000.0
        if end - start <= window_us:
            return (start, end)
        if self.auto_scroll.isChecked():
            return (end - window_us, end)
        return self.scrollbar_xlim(start, end, window_us)

    def update_plot_scrollbar(self, start_us: float, end_us: float, window_us: float, *, follow: bool) -> None:
        self.scroll_data_start_us = start_us
        self.scroll_data_end_us = end_us
        self.scroll_window_us = window_us
        total_us = max(0.0, end_us - start_us)
        self.updating_scrollbar = True
        try:
            if total_us <= 0.0 or window_us <= 0.0 or total_us <= window_us:
                self.plot_scrollbar.setRange(0, 0)
                self.plot_scrollbar.setValue(0)
            else:
                self.plot_scrollbar.setRange(0, SCROLL_STEPS)
                page_step = max(1, int(SCROLL_STEPS * min(1.0, window_us / total_us)))
                self.plot_scrollbar.setPageStep(page_step)
                self.plot_scrollbar.setSingleStep(max(1, page_step // 10))
                if follow:
                    self.plot_scrollbar.setValue(SCROLL_STEPS)
        finally:
            self.updating_scrollbar = False

    def scrollbar_xlim(self, start_us: float, end_us: float, window_us: float) -> tuple[float, float] | None:
        total_us = max(0.0, end_us - start_us)
        if total_us <= 0.0:
            return None
        if window_us <= 0.0 or total_us <= window_us:
            return (start_us, end_us)
        fraction = self.plot_scrollbar.value() / max(1, self.plot_scrollbar.maximum())
        left = start_us + (total_us - window_us) * fraction
        return (left, left + window_us)

    def plot_scrollbar_changed(self, _value: int) -> None:
        if self.updating_scrollbar:
            return
        if self.live_time_us.size:
            self.plot_live()
        elif self.capture is not None:
            self.plot_capture_with_scroll(self.capture)

    def plot_scrollbar_pressed(self) -> None:
        if self.live_time_us.size and self.auto_scroll.isChecked():
            self.auto_scroll.setChecked(False)

    def update_measurements(self, capture: ScopeCapture, elapsed: float) -> None:
        m = measure_scope(capture.raw, capture.sample_rate)
        freq = m["frequency"]
        freq_text = "n/a" if freq is None else f"{freq:.3f} Hz"
        lines = [
            f"Capture time: {elapsed:.3f} s",
            f"Input: ADC{capture.channel} / GPIO{capture.gpio}",
            f"Sample rate: {capture.sample_rate / 1000:.3f} kS/s",
            f"Sample interval: {capture.sample_interval_us:.3f} us",
            f"Window: {capture.samples * capture.sample_interval_us:.3f} us",
            f"Trigger: {capture.trigger_mode} {capture.trigger_slope} at {capture.trigger_level} counts",
            "",
            f"Min: {m['min']:.4f} V",
            f"Max: {m['max']:.4f} V",
            f"Mean: {m['mean']:.4f} V",
            f"Peak-to-peak: {m['pp']:.4f} V",
            f"Estimated frequency: {freq_text}",
            f"Threshold crossings: {m['edges']}",
        ]
        self.measurements.setPlainText("\n".join(lines))

    def save_csv(self) -> None:
        if self.capture is None and self.live_time_us.size == 0:
            QtWidgets.QMessageBox.information(self, "No capture", "Run a capture first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save capture",
            "pico_scope_capture.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_us", "voltage_v", "raw_adc"])
            if self.live_time_us.size:
                raw = np.clip(np.rint(self.live_volts * 4095.0 / 3.3), 0, 4095).astype(int)
                for index, t in enumerate(self.live_time_us):
                    writer.writerow([t, self.live_volts[index], raw[index]])
            elif self.capture is not None:
                volts = self.capture.volts
                for index, t in enumerate(self.capture.time_us):
                    writer.writerow([t, volts[index], int(self.capture.raw[index])])
        self._show_status(f"Saved {path}")

    def _display_changed(self, *_args) -> None:
        if self.live_time_us.size:
            self.plot_live()
        elif self.capture is not None:
            self.plot_capture_with_scroll(self.capture)

    def _update_gpio_label(self) -> None:
        self.gpio_label.setText(f"GPIO{26 + self.channel.value()}")

    def _update_window_label(self) -> None:
        window_ms = self.samples.value() / self.sample_rate.value() * 1000.0
        self.window_label.setText(f"{window_ms:.3f} ms")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")

    def _set_live_button_text(self) -> None:
        self.live_button.setText("Stop Live" if self.live_enabled else "Start Live")

    def _show_status(self, message: str) -> None:
        self.status.showMessage(message)
        self.status_label.setText(message)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
