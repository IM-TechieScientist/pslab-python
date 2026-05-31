#!/usr/bin/env python3
"""Development GUI for the PSLab Pico logic analyzer."""

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

from pslab.instrument.pico_logic_analyzer import LogicCapture, PicoLogicAnalyzer


SYS_CLOCK_HZ = 150_000_000
SCROLL_STEPS = 10000


@dataclass
class CaptureSettings:
    port: str | None
    sys_clock_hz: int
    pin_base: int
    pin_count: int
    samples: int
    divider: int
    trigger_pin: int
    trigger_level: int
    trigger_mode: str


def measure_channel(states: np.ndarray, sample_rate: float) -> dict[str, float | int | None]:
    """Return simple timing measurements for one boolean channel."""

    if states.size < 2:
        return {"edges": 0, "frequency": None, "duty": None}

    edges = np.flatnonzero(states[1:] != states[:-1]) + 1
    if edges.size < 2:
        return {"edges": int(edges.size), "frequency": None, "duty": float(np.mean(states) * 100.0)}

    rising = edges[states[edges]]
    frequency = None
    if rising.size >= 2:
        periods = np.diff(rising) / sample_rate
        mean_period = float(np.mean(periods))
        if mean_period > 0.0:
            frequency = 1.0 / mean_period

    duty = float(np.mean(states) * 100.0)
    return {"edges": int(edges.size), "frequency": frequency, "duty": duty}


class CaptureWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, float)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings: CaptureSettings):
        super().__init__()
        self.settings = settings

    @QtCore.pyqtSlot()
    def run(self) -> None:
        start = time.perf_counter()
        try:
            analyzer = PicoLogicAnalyzer(
                port=self.settings.port,
                timeout=2.0,
                sys_clock_hz=self.settings.sys_clock_hz,
            )
            capture = analyzer.capture(
                pin_base=self.settings.pin_base,
                pin_count=self.settings.pin_count,
                samples=self.settings.samples,
                divider=self.settings.divider,
                trigger_pin=self.settings.trigger_pin,
                trigger_level=self.settings.trigger_level,
                trigger_mode=self.settings.trigger_mode,
            )
            analyzer.disconnect()
        except Exception as exc:  # pragma: no cover - GUI error path
            self.failed.emit(str(exc))
            return

        self.finished.emit(capture, time.perf_counter() - start)


class StreamWorker(QtCore.QObject):
    frame = QtCore.pyqtSignal(int, object, float)
    failed = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, settings: CaptureSettings):
        super().__init__()
        self.settings = settings
        self._stop_requested = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        analyzer = PicoLogicAnalyzer(
            port=self.settings.port,
            timeout=2.0,
            sys_clock_hz=self.settings.sys_clock_hz,
        )
        try:
            analyzer.configure(
                pin_base=self.settings.pin_base,
                pin_count=self.settings.pin_count,
                samples=self.settings.samples,
                divider=self.settings.divider,
                trigger_pin=self.settings.trigger_pin,
                trigger_level=self.settings.trigger_level,
                trigger_mode=self.settings.trigger_mode,
            )
            analyzer.stream_start()
            while not self._stop_requested:
                start = time.perf_counter()
                sequence, capture = analyzer.read_stream_frame()
                self.frame.emit(sequence, capture, time.perf_counter() - start)
        except Exception as exc:  # pragma: no cover - GUI error path
            if not self._stop_requested:
                self.failed.emit(str(exc))
        finally:
            try:
                analyzer.stream_stop()
            except Exception:
                pass
            analyzer.disconnect()
            self.stopped.emit()

    def request_stop(self) -> None:
        self._stop_requested = True


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def plot_capture(self, capture: LogicCapture | None) -> None:
        self.axes.clear()

        if capture is None:
            self._plot_empty("No capture")
            return

        self.plot_states(
            time_us=capture.time_us,
            states=capture.states,
            pin_base=capture.pin_base,
            sample_rate=capture.sample_rate,
            title=(
                f"{capture.sample_rate / 1e6:.3f} MS/s, "
                f"{capture.samples} samples, divider {capture.divider}"
            ),
        )

    def plot_states(
        self,
        *,
        time_us: np.ndarray,
        states: np.ndarray,
        pin_base: int,
        sample_rate: float,
        title: str,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        self.axes.clear()

        if states.size == 0 or time_us.size == 0:
            self._plot_empty("No data")
            return

        spacing = 2
        sample_interval_us = 1e6 / sample_rate
        edge_time = np.concatenate([time_us, [time_us[-1] + sample_interval_us]])
        x = np.repeat(edge_time, 2)[1:-1]

        for index, channel in enumerate(states):
            y = np.repeat(channel.astype(int), 2) + index * spacing
            gpio = pin_base + index
            self.axes.plot(x, y, drawstyle="steps-post", linewidth=1.2, label=f"GPIO{gpio}")

        self.axes.set_xlabel("Time (us)")
        self.axes.set_yticks([i * spacing for i in range(states.shape[0])])
        self.axes.set_yticklabels([f"GPIO{pin_base + i}" for i in range(states.shape[0])])
        self.axes.grid(True, axis="x", alpha=0.25)
        self.axes.legend(loc="upper right")
        self.axes.set_title(title)
        if xlim is not None and xlim[1] > xlim[0]:
            self.axes.set_xlim(*xlim)
        self.draw_idle()

    def _plot_empty(self, title: str) -> None:
        self.axes.set_xlabel("Time (us)")
        self.axes.set_yticks([])
        self.axes.set_title(title)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSLab Pico Logic Analyzer")
        self.resize(1500, 920)

        self.analyzer: PicoLogicAnalyzer | None = None
        self.capture: LogicCapture | None = None
        self.worker_thread: QtCore.QThread | None = None
        self.worker: CaptureWorker | None = None
        self.stream_thread: QtCore.QThread | None = None
        self.stream_worker: StreamWorker | None = None
        self.live_enabled = False
        self.live_time_us = np.array([], dtype=float)
        self.live_states: np.ndarray | None = None
        self.live_pin_base = 0
        self.live_sample_rate = 1.0
        self.scroll_data_start_us = 0.0
        self.scroll_data_end_us = 0.0
        self.scroll_window_us = 0.0
        self.updating_scrollbar = False

        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setInterval(100)
        self.live_timer.timeout.connect(self._live_tick)
        QtWidgets.QShortcut("F11", self, activated=self.toggle_fullscreen)

        self._apply_style()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._top_bar())

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, stretch=1)

        controls = self._settings_tabs()
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setMinimumWidth(320)
        controls_scroll.setMaximumWidth(520)
        controls_scroll.setWidget(controls)
        self.splitter.addWidget(controls_scroll)

        plot_area = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        self.plot = PlotCanvas()
        self.toolbar = NavigationToolbar(self.plot, self)
        self.plot_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.plot_scrollbar.setRange(0, 0)
        self.plot_scrollbar.setSingleStep(25)
        self.plot_scrollbar.setPageStep(1000)
        self.plot_scrollbar.valueChanged.connect(self.plot_scrollbar_changed)
        self.plot_scrollbar.sliderPressed.connect(self.plot_scrollbar_pressed)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.plot, stretch=1)
        plot_layout.addWidget(self.plot_scrollbar)
        self.splitter.addWidget(plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([380, 1120])

        self.status = self.statusBar()
        self._show_status("Ready")
        self._build_menu()

        self.refresh_ports()
        self._update_rate_label()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                font-size: 10pt;
            }
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
            QPushButton {
                min-height: 28px;
                padding: 4px 10px;
            }
            QPushButton[primary="true"] {
                font-weight: 700;
                min-height: 34px;
            }
            QLabel#BrandLabel {
                font-size: 15pt;
                font-weight: 800;
            }
            QLabel#StatusPill {
                border: 1px solid #c9ced6;
                border-radius: 12px;
                padding: 4px 10px;
                background: #f5f7fa;
            }
            QTabWidget::pane {
                border: 1px solid #c9ced6;
                border-radius: 6px;
                top: -1px;
            }
            QPlainTextEdit {
                font-family: monospace;
            }
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
        subtitle = QtWidgets.QLabel("Logic Analyzer")
        subtitle.setStyleSheet("color: #586171;")

        title_box = QtWidgets.QVBoxLayout()
        title_box.setContentsMargins(0, 0, 14, 0)
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)

        self.top_capture_button = QtWidgets.QPushButton("Capture")
        self.top_live_button = QtWidgets.QPushButton("Start Live")
        self.top_clear_button = QtWidgets.QPushButton("Clear")
        self.top_save_button = QtWidgets.QPushButton("Save CSV")
        self.top_fullscreen_button = QtWidgets.QPushButton("Full Screen")
        self.top_status_label = QtWidgets.QLabel("Ready")
        self.top_status_label.setObjectName("StatusPill")

        for button in (self.top_capture_button, self.top_live_button):
            button.setProperty("primary", True)

        self.top_capture_button.clicked.connect(self.start_capture)
        self.top_live_button.clicked.connect(self.toggle_live)
        self.top_clear_button.clicked.connect(self.clear_display)
        self.top_save_button.clicked.connect(self.save_csv)
        self.top_fullscreen_button.clicked.connect(self.toggle_fullscreen)

        layout.addWidget(self.top_capture_button)
        layout.addWidget(self.top_live_button)
        layout.addWidget(self.top_clear_button)
        layout.addWidget(self.top_save_button)
        layout.addStretch(1)
        layout.addWidget(self.top_status_label)
        layout.addWidget(self.top_fullscreen_button)
        return bar

    def _settings_tabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        setup = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(setup)
        setup_layout.setContentsMargins(8, 8, 8, 8)
        setup_layout.setSpacing(10)
        setup_layout.addWidget(self._connection_group())
        setup_layout.addWidget(self._capture_group())
        setup_layout.addWidget(self._display_group())
        setup_layout.addStretch(1)

        tools = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout(tools)
        tools_layout.setContentsMargins(8, 8, 8, 8)
        tools_layout.setSpacing(10)
        tools_layout.addWidget(self._test_signal_group())
        tools_layout.addWidget(self._actions_group())
        tools_layout.addStretch(1)

        measurements = QtWidgets.QWidget()
        measurements_layout = QtWidgets.QVBoxLayout(measurements)
        measurements_layout.setContentsMargins(8, 8, 8, 8)
        measurements_layout.addWidget(self._measurements_group())

        tabs.addTab(setup, "Setup")
        tabs.addTab(tools, "Tools")
        tabs.addTab(measurements, "Measure")
        return tabs

    def _build_menu(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        fullscreen = QtWidgets.QAction("Toggle Full Screen", self)
        fullscreen.setShortcut("F11")
        fullscreen.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen)

        maximize = QtWidgets.QAction("Maximize", self)
        maximize.triggered.connect(self.showMaximized)
        view_menu.addAction(maximize)

    def _connection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Connection")
        layout = QtWidgets.QGridLayout(group)

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.idn_label = QtWidgets.QLabel("Not connected")
        self.idn_label.setWordWrap(True)

        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.identify_device)

        layout.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        layout.addWidget(self.port_combo, 0, 1, 1, 2)
        layout.addWidget(self.refresh_button, 1, 1)
        layout.addWidget(self.connect_button, 1, 2)
        layout.addWidget(self.idn_label, 2, 0, 1, 3)
        return group

    def _capture_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Capture")
        layout = QtWidgets.QGridLayout(group)

        self.sys_clock = self._spin(1_000_000, 500_000_000, SYS_CLOCK_HZ, 1_000_000)
        self.pin_base = self._spin(0, 29, 16)
        self.pin_count = self._spin(1, 8, 1)
        self.samples = self._spin(1, 65536, 4096, 256)
        self.divider = self._spin(1, 1_000_000, 15000, 1)
        self.trigger_pin = self._spin(0, 29, 16)
        self.trigger_level = QtWidgets.QComboBox()
        self.trigger_level.addItems(["1", "0"])
        self.trigger_mode = QtWidgets.QComboBox()
        self.trigger_mode.addItems(["edge", "level"])
        self.rate_label = QtWidgets.QLabel()

        for spin in (self.sys_clock, self.divider):
            spin.valueChanged.connect(self._update_rate_label)

        rows = [
            ("Sys clock", self.sys_clock),
            ("Pin base", self.pin_base),
            ("Pin count", self.pin_count),
            ("Samples", self.samples),
            ("Divider", self.divider),
            ("Sample rate", self.rate_label),
            ("Trigger pin", self.trigger_pin),
            ("Trigger level", self.trigger_level),
            ("Trigger mode", self.trigger_mode),
        ]

        for row, (label, widget) in enumerate(rows):
            layout.addWidget(QtWidgets.QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)

        return group

    def _test_signal_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Built-In Test Signal")
        layout = QtWidgets.QGridLayout(group)

        self.test_pin = self._spin(0, 29, 15)
        self.test_frequency = self._spin(1, 20_000_000, 1000, 1000)
        self.test_start = QtWidgets.QPushButton("Start")
        self.test_stop = QtWidgets.QPushButton("Stop")

        self.test_start.clicked.connect(self.start_test_signal)
        self.test_stop.clicked.connect(self.stop_test_signal)

        layout.addWidget(QtWidgets.QLabel("Pin"), 0, 0)
        layout.addWidget(self.test_pin, 0, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Frequency"), 1, 0)
        layout.addWidget(self.test_frequency, 1, 1, 1, 2)
        layout.addWidget(self.test_start, 2, 1)
        layout.addWidget(self.test_stop, 2, 2)
        return group

    def _actions_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Actions")
        layout = QtWidgets.QGridLayout(group)

        self.capture_button = QtWidgets.QPushButton("Capture")
        self.live_button = QtWidgets.QPushButton("Start Live")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.save_button = QtWidgets.QPushButton("Save CSV")

        self.capture_button.clicked.connect(self.start_capture)
        self.live_button.clicked.connect(self.toggle_live)
        self.clear_button.clicked.connect(self.clear_display)
        self.save_button.clicked.connect(self.save_csv)

        layout.addWidget(self.capture_button, 0, 0)
        layout.addWidget(self.live_button, 0, 1)
        layout.addWidget(self.clear_button, 1, 0)
        layout.addWidget(self.save_button, 1, 1)
        return group

    def _display_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Display")
        layout = QtWidgets.QGridLayout(group)

        self.auto_scroll = QtWidgets.QCheckBox("Auto-scroll live")
        self.auto_scroll.setChecked(True)
        self.live_window_ms = self._spin(1, 60000, 250, 50)
        self.retain_seconds = self._spin(1, 600, 10, 1)
        self.fullscreen_button = QtWidgets.QPushButton("Full Screen")

        self.auto_scroll.stateChanged.connect(self._display_changed)
        self.live_window_ms.valueChanged.connect(self._display_changed)
        self.retain_seconds.valueChanged.connect(self._trim_live_history)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        layout.addWidget(self.auto_scroll, 0, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Live window"), 1, 0)
        layout.addWidget(self.live_window_ms, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Retain s"), 2, 0)
        layout.addWidget(self.retain_seconds, 2, 1)
        layout.addWidget(self.fullscreen_button, 3, 0, 1, 2)
        return group

    def _measurements_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Measurements")
        layout = QtWidgets.QVBoxLayout(group)
        self.measurements = QtWidgets.QPlainTextEdit()
        self.measurements.setReadOnly(True)
        self.measurements.setMinimumHeight(160)
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
            label = f"{port.device}"
            if port.product:
                label += f" - {port.product}"
            self.port_combo.addItem(label, port.device)
        if selected:
            index = self.port_combo.findData(selected)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

    def identify_device(self) -> None:
        try:
            analyzer = PicoLogicAnalyzer(
                port=self.port_combo.currentData(),
                sys_clock_hz=self.sys_clock.value(),
            )
            identity = analyzer.identify()
            analyzer.disconnect()
        except Exception as exc:
            self.idn_label.setText(f"Connection failed: {exc}")
            self._show_status("Connection failed")
            return

        self.idn_label.setText(identity)
        self._show_status("Connected")

    def current_settings(self) -> CaptureSettings:
        return CaptureSettings(
            port=self.port_combo.currentData(),
            sys_clock_hz=self.sys_clock.value(),
            pin_base=self.pin_base.value(),
            pin_count=self.pin_count.value(),
            samples=self.samples.value(),
            divider=self.divider.value(),
            trigger_pin=self.trigger_pin.value(),
            trigger_level=int(self.trigger_level.currentText()),
            trigger_mode=self.trigger_mode.currentText(),
        )

    def start_capture(self) -> None:
        if self.worker_thread is not None:
            return

        settings = self.current_settings()
        self.capture_button.setEnabled(False)
        self.top_capture_button.setEnabled(False)
        self._show_status("Capturing...")

        self.worker_thread = QtCore.QThread(self)
        self.worker = CaptureWorker(settings)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.capture_finished)
        self.worker.failed.connect(self.capture_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.worker_done)
        self.worker_thread.start()

    def capture_finished(self, capture: LogicCapture, elapsed: float) -> None:
        self.capture = capture
        if self.live_enabled:
            self.append_live_capture(capture)
            self.plot_live()
        else:
            self.plot_capture_with_scroll(capture)
        self.update_measurements(capture, elapsed)
        self._show_status(f"Capture complete in {elapsed:.3f} s")

    def capture_failed(self, message: str) -> None:
        self._show_status(f"Capture failed: {message}")
        if self.live_enabled:
            self.toggle_live()
        QtWidgets.QMessageBox.warning(self, "Capture failed", message)

    def worker_done(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.capture_button.setEnabled(True)
        self.top_capture_button.setEnabled(True)

    def toggle_live(self) -> None:
        if self.live_enabled:
            self.stop_stream()
        else:
            self.start_stream()

    def _live_tick(self) -> None:
        if self.live_enabled and self.worker_thread is None:
            self.start_capture()

    def start_stream(self) -> None:
        if self.stream_thread is not None:
            return

        self.clear_live_history()
        self.live_enabled = True
        self._set_live_button_text()
        self.top_capture_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self._show_status("Starting stream...")

        self.stream_thread = QtCore.QThread(self)
        self.stream_worker = StreamWorker(self.current_settings())
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
        self.live_timer.stop()
        if self.stream_worker is not None:
            self.stream_worker.request_stop()
            self._show_status("Stopping stream...")
        else:
            self._show_status("Live capture stopped")

    def stream_frame_received(self, sequence: int, capture: LogicCapture, elapsed: float) -> None:
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
        self.top_capture_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        self._show_status("Live capture stopped")

    def append_live_capture(self, capture: LogicCapture) -> None:
        sample_interval_us = capture.sample_interval_us
        if (
            self.live_states is None
            or self.live_states.shape[0] != capture.pin_count
            or self.live_pin_base != capture.pin_base
            or abs(self.live_sample_rate - capture.sample_rate) > 0.5
        ):
            self.clear_live_history()
            self.live_pin_base = capture.pin_base
            self.live_sample_rate = capture.sample_rate

        start_us = 0.0
        if self.live_time_us.size:
            start_us = float(self.live_time_us[-1] + sample_interval_us)

        new_time = start_us + np.arange(capture.samples) * sample_interval_us
        if self.live_states is None:
            self.live_states = capture.states.copy()
        else:
            self.live_states = np.concatenate([self.live_states, capture.states], axis=1)
        self.live_time_us = np.concatenate([self.live_time_us, new_time])
        self._trim_live_history()

    def clear_live_history(self) -> None:
        self.live_time_us = np.array([], dtype=float)
        self.live_states = None
        self.live_sample_rate = max(1.0, self.sys_clock.value() / self.divider.value())
        self.live_pin_base = self.pin_base.value()
        self.update_plot_scrollbar(0.0, 0.0, 0.0, follow=True)

    def clear_display(self) -> None:
        self.capture = None
        self.clear_live_history()
        self.measurements.clear()
        self.plot.plot_capture(None)
        self._show_status("Display cleared")

    def _trim_live_history(self, *_args) -> None:
        if self.live_states is None or self.live_time_us.size == 0:
            return

        retain_us = self.retain_seconds.value() * 1_000_000.0
        cutoff = self.live_time_us[-1] - retain_us
        first = int(np.searchsorted(self.live_time_us, cutoff, side="left"))
        if first > 0:
            self.live_time_us = self.live_time_us[first:]
            self.live_states = self.live_states[:, first:]
        self.plot_live()

    def plot_live(self) -> None:
        if self.live_states is None or self.live_time_us.size == 0:
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
        self.plot.plot_states(
            time_us=self.live_time_us,
            states=self.live_states,
            pin_base=self.live_pin_base,
            sample_rate=self.live_sample_rate,
            title=(
                f"Live history, {self.live_sample_rate / 1e6:.3f} MS/s, "
                f"{self.live_time_us.size} retained samples"
            ),
            xlim=xlim,
        )

    def plot_capture_with_scroll(self, capture: LogicCapture) -> None:
        start = 0.0
        end = float(capture.time_us[-1]) if capture.time_us.size else 0.0
        window = min(end - start, self.live_window_ms.value() * 1000.0)
        if window <= 0.0:
            window = end - start

        self.update_plot_scrollbar(start, end, window, follow=False)
        xlim = self.scrollbar_xlim(start, end, window)
        self.plot.plot_states(
            time_us=capture.time_us,
            states=capture.states,
            pin_base=capture.pin_base,
            sample_rate=capture.sample_rate,
            title=(
                f"{capture.sample_rate / 1e6:.3f} MS/s, "
                f"{capture.samples} samples, divider {capture.divider}"
            ),
            xlim=xlim,
        )

    def _live_xlim(self) -> tuple[float, float] | None:
        if self.live_time_us.size == 0:
            return None

        start = float(self.live_time_us[0])
        end = float(self.live_time_us[-1])
        window_us = self.live_window_ms.value() * 1000.0
        if end - start <= window_us:
            return (start, end)

        if self.auto_scroll.isChecked():
            return (end - window_us, end)

        return self.scrollbar_xlim(start, end, window_us)

    def _display_changed(self, *_args) -> None:
        if self.live_states is not None:
            self.plot_live()
        elif self.capture is not None:
            self.plot_capture_with_scroll(self.capture)

    def update_plot_scrollbar(
        self,
        start_us: float,
        end_us: float,
        window_us: float,
        *,
        follow: bool,
    ) -> None:
        self.scroll_data_start_us = start_us
        self.scroll_data_end_us = end_us
        self.scroll_window_us = window_us

        total_us = max(0.0, end_us - start_us)
        self.updating_scrollbar = True
        try:
            if total_us <= 0.0 or window_us <= 0.0 or total_us <= window_us:
                self.plot_scrollbar.setRange(0, 0)
                self.plot_scrollbar.setPageStep(SCROLL_STEPS)
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

    def scrollbar_xlim(
        self,
        start_us: float,
        end_us: float,
        window_us: float,
    ) -> tuple[float, float] | None:
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
        if self.live_states is not None:
            self.plot_live()
        elif self.capture is not None:
            self.plot_capture_with_scroll(self.capture)

    def plot_scrollbar_pressed(self) -> None:
        if self.live_states is not None and self.auto_scroll.isChecked():
            self.auto_scroll.setChecked(False)

    def update_measurements(self, capture: LogicCapture, elapsed: float) -> None:
        lines = [
            f"Capture time: {elapsed:.3f} s",
            f"Sample rate: {capture.sample_rate / 1e6:.6f} MS/s",
            f"Sample interval: {capture.sample_interval_us:.6f} us",
            f"Window: {capture.samples * capture.sample_interval_us:.3f} us",
            "",
        ]

        for index, states in enumerate(capture.states):
            gpio = capture.pin_base + index
            measurement = measure_channel(states, capture.sample_rate)
            frequency = measurement["frequency"]
            frequency_text = "n/a" if frequency is None else f"{frequency:.3f} Hz"
            duty = measurement["duty"]
            duty_text = "n/a" if duty is None else f"{duty:.2f}%"
            lines.append(
                f"GPIO{gpio}: edges={measurement['edges']} "
                f"frequency={frequency_text} duty={duty_text}"
            )

        self.measurements.setPlainText("\n".join(lines))

    def start_test_signal(self) -> None:
        try:
            analyzer = PicoLogicAnalyzer(
                port=self.port_combo.currentData(),
                sys_clock_hz=self.sys_clock.value(),
            )
            analyzer.start_test_square(self.test_pin.value(), self.test_frequency.value())
            analyzer.disconnect()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test signal failed", str(exc))
            return
        self._show_status("Built-in test signal started")

    def stop_test_signal(self) -> None:
        try:
            analyzer = PicoLogicAnalyzer(
                port=self.port_combo.currentData(),
                sys_clock_hz=self.sys_clock.value(),
            )
            analyzer.stop_test_square()
            analyzer.disconnect()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test signal failed", str(exc))
            return
        self._show_status("Built-in test signal stopped")

    def save_csv(self) -> None:
        if self.capture is None and self.live_states is None:
            QtWidgets.QMessageBox.information(self, "No capture", "Run a capture first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save capture",
            "pico_logic_capture.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if self.live_states is not None and self.live_time_us.size:
                header = ["time_us"] + [
                    f"GPIO{self.live_pin_base + index}"
                    for index in range(self.live_states.shape[0])
                ]
                writer.writerow(header)
                for sample in range(self.live_time_us.size):
                    writer.writerow(
                        [self.live_time_us[sample]]
                        + [int(self.live_states[index, sample]) for index in range(self.live_states.shape[0])]
                    )
            elif self.capture is not None:
                header = ["time_us"] + [
                    f"GPIO{self.capture.pin_base + index}"
                    for index in range(self.capture.pin_count)
                ]
                writer.writerow(header)
                for sample in range(self.capture.samples):
                    writer.writerow(
                        [self.capture.time_us[sample]]
                        + [int(self.capture.states[index, sample]) for index in range(self.capture.pin_count)]
                    )

        self._show_status(f"Saved {path}")

    def _update_rate_label(self) -> None:
        rate = self.sys_clock.value() / self.divider.value()
        self.rate_label.setText(f"{rate / 1e6:.6f} MS/s")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("Full Screen")
            self.top_fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")
            self.top_fullscreen_button.setText("Exit Full Screen")

    def _set_live_button_text(self) -> None:
        text = "Stop Live" if self.live_enabled else "Start Live"
        self.live_button.setText(text)
        self.top_live_button.setText(text)

    def _show_status(self, message: str) -> None:
        self.status.showMessage(message)
        self.top_status_label.setText(message)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
