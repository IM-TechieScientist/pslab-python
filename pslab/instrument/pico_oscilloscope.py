"""SCPI client for the PSLab Pico oscilloscope firmware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pslab.instrument.pico_logic_analyzer import PicoLogicAnalyzer, ScpiError


@dataclass
class ScopeCapture:
    """Captured ADC samples from the Pico DSO instrument."""

    channel: int
    gpio: int
    sample_rate: float
    samples: int
    trigger_level: int
    trigger_mode: str
    trigger_slope: str
    raw: np.ndarray

    @property
    def sample_interval_us(self) -> float:
        return 1e6 / self.sample_rate

    @property
    def time_us(self) -> np.ndarray:
        return np.arange(self.samples) * self.sample_interval_us

    @property
    def volts(self) -> np.ndarray:
        return self.raw.astype(np.float64) * (3.3 / 4095.0)


@dataclass
class ScopeStreamStatus:
    enabled: bool
    sequence: int
    overruns: int


class PicoOscilloscope(PicoLogicAnalyzer):
    """Control the PSLab Pico rudimentary ADC oscilloscope over USB CDC."""

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 2.0,
        serial_instance=None,
    ):
        super().__init__(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            serial_instance=serial_instance,
        )
        self.channel = 0
        self.gpio = 26
        self.sample_rate = 100_000
        self.samples = 1024
        self.trigger_level = 2048
        self.trigger_mode = "OFF"
        self.trigger_slope = "RISE"
        self.last_scope_capture: ScopeCapture | None = None

    def configure(
        self,
        *,
        channel: int | None = None,
        sample_rate: int | None = None,
        samples: int | None = None,
        trigger_level: int | None = None,
        trigger_mode: str | None = None,
        trigger_slope: str | None = None,
    ) -> None:
        if channel is not None:
            self.command(f"DSO:CONF:CHAN {channel}")
            self.channel = int(channel)
            self.gpio = int(self.query("DSO:CONF:GPIO?"))
        if sample_rate is not None:
            self.command(f"DSO:CONF:RATE {sample_rate}")
            self.sample_rate = int(sample_rate)
        if samples is not None:
            self.command(f"DSO:CONF:SAMP {samples}")
            self.samples = int(samples)
        if trigger_level is not None:
            self.command(f"DSO:CONF:TRIG:LEV {trigger_level}")
            self.trigger_level = int(trigger_level)
        if trigger_mode is not None:
            mode = self._normalize_trigger_mode_dso(trigger_mode)
            self.command(f"DSO:CONF:TRIG:MODE {mode}")
            self.trigger_mode = mode
        if trigger_slope is not None:
            slope = self._normalize_trigger_slope(trigger_slope)
            self.command(f"DSO:CONF:TRIG:SLOP {slope}")
            self.trigger_slope = slope

    def refresh_configuration(self) -> None:
        self.channel = int(self.query("DSO:CONF:CHAN?"))
        self.gpio = int(self.query("DSO:CONF:GPIO?"))
        self.sample_rate = int(self.query("DSO:CONF:RATE?"))
        self.samples = int(self.query("DSO:CONF:SAMP?"))
        self.trigger_level = int(self.query("DSO:CONF:TRIG:LEV?"))
        self.trigger_mode = self.query("DSO:CONF:TRIG:MODE?").upper()
        self.trigger_slope = self.query("DSO:CONF:TRIG:SLOP?").upper()

    def initiate(self) -> None:
        self.command("DSO:INIT", timeout=None)

    def fetch(self) -> ScopeCapture:
        payload = self.query_block("DSO:FETC?")
        return self._capture_from_scope_payload(payload)

    def capture(
        self,
        *,
        channel: int | None = None,
        sample_rate: int | None = None,
        samples: int | None = None,
        trigger_level: int | None = None,
        trigger_mode: str | None = None,
        trigger_slope: str | None = None,
    ) -> ScopeCapture:
        self.configure(
            channel=channel,
            sample_rate=sample_rate,
            samples=samples,
            trigger_level=trigger_level,
            trigger_mode=trigger_mode,
            trigger_slope=trigger_slope,
        )
        payload = self.query_block("DSO:READ?", timeout=None)
        return self._capture_from_scope_payload(payload)

    def stream_start(self) -> None:
        self.command("DSO:STREAM:START")

    def stream_stop(self) -> None:
        self._write_line("DSO:STREAM:STOP")
        while True:
            line = self._read_line()
            if line == "OK":
                return
            if line.startswith("DSO:STREAM:FRAME "):
                self._read_block_payload()
                continue
            if line == "DSO:STREAM:ERROR":
                continue
            raise ScpiError(line)

    def stream_status(self) -> ScopeStreamStatus:
        fields = self.query("DSO:STREAM:STAT?").split(",")
        if len(fields) != 3:
            raise ScpiError(f"Unexpected DSO stream status: {fields!r}")
        return ScopeStreamStatus(
            enabled=bool(int(fields[0])),
            sequence=int(fields[1]),
            overruns=int(fields[2]),
        )

    def read_stream_frame(self) -> tuple[int, ScopeCapture]:
        line = self._read_line()
        fields = line.split()
        if len(fields) != 3 or fields[0] != "DSO:STREAM:FRAME":
            raise ScpiError(f"Unexpected DSO stream frame header: {line!r}")

        sequence = int(fields[1])
        payload_len = int(fields[2])
        payload = self._read_block_payload()
        if len(payload) != payload_len:
            raise ScpiError(
                f"DSO stream payload length mismatch: header={payload_len}, got={len(payload)}"
            )
        return sequence, self._capture_from_scope_payload(payload)

    def _capture_from_scope_payload(self, payload: bytes) -> ScopeCapture:
        if len(payload) % 2:
            raise ScpiError(f"DSO payload length is not uint16 aligned: {len(payload)}")

        raw = np.frombuffer(payload, dtype="<u2").copy()
        capture = ScopeCapture(
            channel=self.channel,
            gpio=self.gpio,
            sample_rate=float(self.sample_rate),
            samples=raw.size,
            trigger_level=self.trigger_level,
            trigger_mode=self.trigger_mode,
            trigger_slope=self.trigger_slope,
            raw=raw,
        )
        self.last_scope_capture = capture
        return capture

    @staticmethod
    def _normalize_trigger_mode_dso(trigger_mode: str) -> str:
        mode = trigger_mode.strip().upper()
        if mode not in ("OFF", "LEVEL", "EDGE"):
            raise ValueError("trigger_mode must be OFF, LEVEL, or EDGE.")
        return mode

    @staticmethod
    def _normalize_trigger_slope(trigger_slope: str) -> str:
        slope = trigger_slope.strip().upper()
        aliases = {
            "RISING": "RISE",
            "RISE": "RISE",
            "FALLING": "FALL",
            "FALL": "FALL",
        }
        if slope not in aliases:
            raise ValueError("trigger_slope must be RISE or FALL.")
        return aliases[slope]
