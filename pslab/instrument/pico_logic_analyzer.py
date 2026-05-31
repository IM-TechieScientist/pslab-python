"""SCPI client for the PSLab Pico firmware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import serial
from serial.tools import list_ports


class ScpiError(RuntimeError):
    """Raised when the Pico logic analyser reports or implies a SCPI error."""


@dataclass
class LogicCapture:
    """Captured sampled logic data.

    Attributes
    ----------
    pin_base:
        First GPIO sampled by the firmware.
    pin_count:
        Number of consecutive GPIO pins captured.
    samples:
        Number of samples per pin.
    divider:
        PIO clock divider used for capture.
    trigger_pin:
        GPIO used for trigger level wait.
    trigger_level:
        Trigger level used for capture.
    trigger_mode:
        Trigger mode, either ``"edge"`` or ``"level"``.
    words:
        Raw little-endian uint32 words returned by the firmware.
    states:
        Boolean array with shape ``(pin_count, samples)``.
    sample_rate:
        Approximate sample rate in samples per second.
    """

    pin_base: int
    pin_count: int
    samples: int
    divider: int
    trigger_pin: int
    trigger_level: bool
    trigger_mode: str
    words: np.ndarray
    states: np.ndarray
    sample_rate: float

    @property
    def sample_interval_us(self) -> float:
        """Approximate time between adjacent samples in microseconds."""

        return 1e6 / self.sample_rate

    @property
    def time_us(self) -> np.ndarray:
        """Sample time axis in microseconds."""

        return np.arange(self.samples) * self.sample_interval_us


@dataclass
class LogicStreamStatus:
    """Firmware streaming status."""

    enabled: bool
    sequence: int
    overruns: int


class PicoLogicAnalyzer:
    """Control the standalone Pico logic analyser firmware over USB CDC.

    This class speaks the SCPI-style protocol implemented by the Pico firmware.
    It is intentionally separate from :class:`pslab.instrument.logic_analyzer`
    because that class targets the classic PSLab binary command protocol and
    returns edge timestamps, while the Pico firmware returns sampled GPIO
    levels.
    """

    USB_VID = 0xCAFE
    USB_PID = 0x4010
    DEFAULT_SYS_CLOCK_HZ = 125_000_000

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 2.0,
        serial_instance=None,
        sys_clock_hz: int = DEFAULT_SYS_CLOCK_HZ,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.sys_clock_hz = sys_clock_hz
        self._ser = serial_instance

        self.pin_base = 16
        self.pin_count = 2
        self.samples = 96
        self.divider = 1
        self.trigger_pin = 16
        self.trigger_level = True
        self.trigger_mode = "edge"
        self.last_capture: LogicCapture | None = None

    @classmethod
    def find_port(cls) -> str:
        """Return the serial port for the Pico logic analyser."""

        for port in list_ports.comports():
            if port.vid == cls.USB_VID and port.pid == cls.USB_PID:
                return port.device
            if port.product and port.product.startswith("PSLab Pico"):
                return port.device

        raise ConnectionError("Pico logic analyser USB CDC port not found.")

    def connect(self) -> None:
        """Open the serial connection if it is not already open."""

        if self._ser is not None and getattr(self._ser, "is_open", True):
            return

        if self.port is None:
            self.port = self.find_port()

        self._ser = serial.Serial(
            self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )

    def disconnect(self) -> None:
        """Close the serial connection."""

        if self._ser is not None:
            self._ser.close()

    def identify(self) -> str:
        """Return the firmware identity string."""

        return self.query("*IDN?")

    def configure(
        self,
        *,
        pin_base: int | None = None,
        pin_count: int | None = None,
        samples: int | None = None,
        divider: int | None = None,
        trigger_pin: int | None = None,
        trigger_level: bool | int | None = None,
        trigger_mode: str | None = None,
    ) -> None:
        """Configure the next capture."""

        if pin_base is not None:
            self.command(f"LA:CONF:PINB {pin_base}")
            self.pin_base = int(pin_base)
        if pin_count is not None:
            self.command(f"LA:CONF:PINC {pin_count}")
            self.pin_count = int(pin_count)
        if samples is not None:
            self.command(f"LA:CONF:SAMP {samples}")
            self.samples = int(samples)
        if divider is not None:
            self.command(f"LA:CONF:DIV {divider}")
            self.divider = int(divider)
        if trigger_pin is not None:
            self.command(f"LA:CONF:TRIG:PIN {trigger_pin}")
            self.trigger_pin = int(trigger_pin)
        if trigger_level is not None:
            level = int(bool(trigger_level))
            self.command(f"LA:CONF:TRIG:LEV {level}")
            self.trigger_level = bool(level)
        if trigger_mode is not None:
            mode = self._normalize_trigger_mode(trigger_mode)
            self.command(f"LA:CONF:TRIG:MODE {mode.upper()}")
            self.trigger_mode = mode

    def refresh_configuration(self) -> None:
        """Read current configuration values back from the firmware."""

        self.pin_base = int(self.query("LA:CONF:PINB?"))
        self.pin_count = int(self.query("LA:CONF:PINC?"))
        self.samples = int(self.query("LA:CONF:SAMP?"))
        self.divider = int(self.query("LA:CONF:DIV?"))
        self.trigger_pin = int(self.query("LA:CONF:TRIG:PIN?"))
        self.trigger_level = bool(int(self.query("LA:CONF:TRIG:LEV?")))
        self.trigger_mode = self.query("LA:CONF:TRIG:MODE?").lower()

    def initiate(self) -> None:
        """Run one blocking capture and leave the data on the device."""

        self.command("LA:INIT", timeout=None)

    def fetch(self) -> LogicCapture:
        """Fetch the most recent capture without starting a new one."""

        payload = self.query_block("LA:FETC?")
        return self._capture_from_payload(payload)

    def stream_start(self) -> None:
        """Start firmware repeated-capture streaming."""

        self.command("LA:STREAM:START")

    def stream_stop(self) -> None:
        """Stop firmware repeated-capture streaming."""

        self._write_line("LA:STREAM:STOP")
        while True:
            line = self._read_line()
            if line == "OK":
                return
            if line.startswith("LA:STREAM:FRAME "):
                self._read_block_payload()
                continue
            if line == "LA:STREAM:ERROR":
                continue
            raise ScpiError(line)

    def stream_status(self) -> LogicStreamStatus:
        """Return streaming status as ``enabled, sequence, overruns``."""

        fields = self.query("LA:STREAM:STAT?").split(",")
        if len(fields) != 3:
            raise ScpiError(f"Unexpected stream status: {fields!r}")
        return LogicStreamStatus(
            enabled=bool(int(fields[0])),
            sequence=int(fields[1]),
            overruns=int(fields[2]),
        )

    def read_stream_frame(self) -> tuple[int, LogicCapture]:
        """Read one frame emitted by ``LA:STREAM:START``.

        The firmware currently streams repeated finite captures. Each frame is a
        text header followed by one SCPI definite-length binary block.
        """

        line = self._read_line()
        fields = line.split()
        if len(fields) != 3 or fields[0] != "LA:STREAM:FRAME":
            raise ScpiError(f"Unexpected stream frame header: {line!r}")

        sequence = int(fields[1])
        payload_len = int(fields[2])
        payload = self._read_block_payload()
        if len(payload) != payload_len:
            raise ScpiError(
                f"Stream payload length mismatch: header={payload_len}, got={len(payload)}"
            )

        return sequence, self._capture_from_payload(payload)

    def capture(
        self,
        *,
        pin_base: int | None = None,
        pin_count: int | None = None,
        samples: int | None = None,
        divider: int | None = None,
        trigger_pin: int | None = None,
        trigger_level: bool | int | None = None,
        trigger_mode: str | None = None,
    ) -> LogicCapture:
        """Configure, capture, fetch, decode, and return sampled logic data."""

        self.configure(
            pin_base=pin_base,
            pin_count=pin_count,
            samples=samples,
            divider=divider,
            trigger_pin=trigger_pin,
            trigger_level=trigger_level,
            trigger_mode=trigger_mode,
        )
        payload = self.query_block("LA:READ?", timeout=None)
        return self._capture_from_payload(payload)

    def get_xy(self, capture: LogicCapture | None = None) -> list[np.ndarray]:
        """Return x/y arrays suitable for step plotting.

        The returned list follows the existing PSLab convention:
        ``[x0, y0, x1, y1, ...]``.
        """

        capture = capture if capture is not None else self.last_capture
        if capture is None:
            raise ValueError("No capture data available.")

        edge_time = np.arange(capture.samples + 1) * capture.sample_interval_us
        x = np.repeat(edge_time, 2)[1:-1]

        xy = []
        for channel in capture.states:
            y = np.repeat(channel.astype(int), 2)
            xy.extend([x.copy(), y])

        return xy

    def plot(self, capture: LogicCapture | None = None, ax=None, spacing: int = 2):
        """Plot the captured waveform using matplotlib."""

        import matplotlib.pyplot as plt

        capture = capture if capture is not None else self.last_capture
        if capture is None:
            raise ValueError("No capture data available.")

        if ax is None:
            _, ax = plt.subplots()

        xy = self.get_xy(capture)
        for channel_index in range(capture.pin_count):
            x = xy[2 * channel_index]
            y = xy[2 * channel_index + 1] + channel_index * spacing
            label = f"GPIO{capture.pin_base + channel_index}"
            ax.plot(x, y, drawstyle="steps-post", label=label)

        ax.set_xlabel("Time (us)")
        ax.set_yticks([i * spacing for i in range(capture.pin_count)])
        ax.set_yticklabels(
            [f"GPIO{capture.pin_base + i}" for i in range(capture.pin_count)]
        )
        ax.legend(loc="upper right")
        return ax

    def start_test_square(self, pin: int = 15, frequency: int = 1000) -> None:
        """Start the firmware's built-in PWM square-wave test output."""

        self.command(f"TEST:SQUARE:CONF {pin} {frequency}")

    def stop_test_square(self) -> None:
        """Stop the firmware's built-in square-wave test output."""

        self.command("TEST:SQUARE OFF")

    def test_square_enabled(self) -> bool:
        """Return whether the built-in square-wave test output is enabled."""

        return bool(int(self.query("TEST:SQUARE?")))

    def command(self, command: str, timeout: float | None = ...):
        """Send a command that should respond with ``OK``."""

        response = self.query(command, timeout=timeout)
        if response != "OK":
            raise ScpiError(response or self.get_error())

    def query(self, command: str, timeout: float | None = ...) -> str:
        """Send a text command and read one line of response."""

        self._write_line(command)
        old_timeout = self._set_timeout(timeout)
        try:
            response = self._ser.readline()
        finally:
            self._restore_timeout(old_timeout)

        if not response:
            raise TimeoutError(f"Timed out waiting for response to {command!r}.")

        return response.decode("ascii", errors="replace").strip()

    def query_block(self, command: str, timeout: float | None = ...) -> bytes:
        """Send a query and read a SCPI arbitrary block response."""

        self._write_line(command)
        return self._read_block_payload(timeout=timeout)

    def _read_block_payload(self, timeout: float | None = ...) -> bytes:
        """Read one SCPI arbitrary block payload."""

        old_timeout = self._set_timeout(timeout)
        try:
            marker = self._read_exact(1)
            if marker != b"#":
                rest = self._ser.readline().decode("ascii", errors="replace").strip()
                raise ScpiError((marker + rest.encode()).decode(errors="replace"))

            digit_count = int(self._read_exact(1).decode("ascii"))
            byte_count = int(self._read_exact(digit_count).decode("ascii"))
            payload = self._read_exact(byte_count)
            self._ser.read(1)
        finally:
            self._restore_timeout(old_timeout)

        return payload

    def get_error(self) -> str:
        """Return and clear the firmware's stored error."""

        return self.query("SYST:ERR?")

    def _capture_from_payload(self, payload: bytes) -> LogicCapture:
        if len(payload) % 4:
            raise ScpiError(f"Capture payload length is not word aligned: {len(payload)}")

        words = np.frombuffer(payload, dtype="<u4").copy()
        states = self.decode_words(words, self.pin_count, self.samples)
        capture = LogicCapture(
            pin_base=self.pin_base,
            pin_count=self.pin_count,
            samples=self.samples,
            divider=self.divider,
            trigger_pin=self.trigger_pin,
            trigger_level=self.trigger_level,
            trigger_mode=self.trigger_mode,
            words=words,
            states=states,
            sample_rate=self.sys_clock_hz / self.divider,
        )
        self.last_capture = capture
        return capture

    @staticmethod
    def decode_words(
        words: Iterable[int],
        pin_count: int,
        samples: int,
    ) -> np.ndarray:
        """Decode packed PIO FIFO words into ``(pin_count, samples)`` states."""

        words = np.asarray(words, dtype=np.uint32)
        bits_per_word = 32 - (32 % pin_count)
        states = np.zeros((pin_count, samples), dtype=bool)

        for pin_offset in range(pin_count):
            for sample_index in range(samples):
                bit_index = pin_offset + sample_index * pin_count
                word_index = bit_index // bits_per_word
                bit_offset = bit_index % bits_per_word + 32 - bits_per_word
                states[pin_offset, sample_index] = bool(
                    words[word_index] & np.uint32(1 << bit_offset)
                )

        return states

    @staticmethod
    def _normalize_trigger_mode(trigger_mode: str) -> str:
        mode = trigger_mode.strip().lower()
        if mode not in ("edge", "level"):
            raise ValueError("trigger_mode must be 'edge' or 'level'.")
        return mode

    def _write_line(self, command: str) -> None:
        self.connect()
        self._ser.write(command.encode("ascii") + b"\n")

    def _read_line(self, timeout: float | None = ...) -> str:
        old_timeout = self._set_timeout(timeout)
        try:
            response = self._ser.readline()
        finally:
            self._restore_timeout(old_timeout)

        if not response:
            raise TimeoutError("Timed out waiting for line response.")

        return response.decode("ascii", errors="replace").strip()

    def _read_exact(self, size: int) -> bytes:
        data = self._ser.read(size)
        if len(data) != size:
            raise TimeoutError(f"Expected {size} bytes, received {len(data)}.")
        return data

    def _set_timeout(self, timeout):
        if timeout is ...:
            return None

        old_timeout = self._ser.timeout
        self._ser.timeout = timeout
        return old_timeout

    def _restore_timeout(self, old_timeout) -> None:
        if old_timeout is not None:
            self._ser.timeout = old_timeout
