from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pslab import PicoLogicAnalyzer


SYSCLK = 150_000_000

# Connect a logic-level CAN signal here, for example CAN_RX from a CAN
# transceiver. Do not connect raw CANH/CANL directly to a Pico GPIO.
CAN_GPIO = 16
CAN_BITRATE = 500_000
SAMPLES_PER_BIT = 20
CAPTURE_BITS = 200
MAX_LA_SAMPLES = 4096


def divider_for_bitrate(bit_rate: int, samples_per_bit: int) -> int:
    target_sample_rate = bit_rate * samples_per_bit
    return max(1, round(SYSCLK / target_sample_rate))


def summarize_can_edges(capture, bit_rate: int) -> None:
    states = capture.states[0].astype(int)
    edge_indexes = np.flatnonzero(states[1:] != states[:-1]) + 1
    edge_times_us = capture.time_us[edge_indexes]
    bit_time_us = 1_000_000 / bit_rate

    print(f"sample_rate={capture.sample_rate / 1e6:.3f} MS/s")
    print(f"sample_interval={capture.sample_interval_us:.3f} us")
    print(f"nominal_CAN_bit_time={bit_time_us:.3f} us")
    print(f"edges_detected={len(edge_indexes)}")

    if len(edge_times_us) < 2:
        print("No CAN-looking transitions found. Check wiring, bitrate, and trigger.")
        return

    edge_spacing_us = np.diff(edge_times_us)
    shortest = float(np.min(edge_spacing_us))
    rough_bitrate = 1_000_000 / shortest

    print(f"shortest_edge_spacing={shortest:.3f} us")
    print(f"rough_bitrate_from_shortest_spacing={rough_bitrate:.0f} bit/s")
    print("first_edges_us=", np.array2string(edge_times_us[:20], precision=2))


def plot_can_capture(capture, bit_rate: int) -> None:
    bit_time_us = 1_000_000 / bit_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    la = PicoLogicAnalyzer(sys_clock_hz=SYSCLK)
    la.plot(capture, ax=ax)

    max_time_us = float(capture.time_us[-1])
    for bit_start_us in np.arange(0, max_time_us, bit_time_us):
        ax.axvline(bit_start_us, color="0.75", linewidth=0.5, alpha=0.35)

    ax.set_title(
        f"CAN capture on GPIO{capture.pin_base} "
        f"({bit_rate / 1000:.0f} kbit/s, {capture.sample_rate / 1e6:.2f} MS/s)"
    )
    ax.set_ylim(-0.4, 1.4)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    plt.show()


def capture_can(
    *,
    can_gpio: int = CAN_GPIO,
    bit_rate: int = CAN_BITRATE,
    samples_per_bit: int = SAMPLES_PER_BIT,
    capture_bits: int = CAPTURE_BITS,
):
    divider = divider_for_bitrate(bit_rate, samples_per_bit)
    sample_rate = SYSCLK / divider
    requested_samples = int(capture_bits * sample_rate / bit_rate)
    samples = min(requested_samples, MAX_LA_SAMPLES)
    actual_capture_bits = samples / (sample_rate / bit_rate)

    print("Waiting for CAN start-of-frame dominant low...")
    print(
        f"gpio={can_gpio}, bitrate={bit_rate}, divider={divider}, "
        f"samples={samples}, window={actual_capture_bits:.1f} bits"
    )
    if requested_samples > MAX_LA_SAMPLES:
        print(
            f"Requested {requested_samples} samples, capped to "
            f"firmware limit {MAX_LA_SAMPLES}."
        )

    la = PicoLogicAnalyzer(sys_clock_hz=SYSCLK)
    try:
        capture = la.capture(
            pin_base=can_gpio,
            pin_count=1,
            samples=samples,
            divider=divider,
            trigger_pin=can_gpio,
            trigger_level=0,
            trigger_mode="edge",
        )
    finally:
        la.disconnect()

    summarize_can_edges(capture, bit_rate)
    plot_can_capture(capture, bit_rate)
    return capture


if __name__ == "__main__":
    capture_can()
