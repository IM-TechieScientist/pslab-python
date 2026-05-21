# from pslab import PicoLogicAnalyzer
# la = PicoLogicAnalyzer()
# print(la.identify())

from pslab import PicoLogicAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import time

SYSCLK = 150_000_000

def estimate_frequency(capture):
    y = capture.states[0].astype(int)
    t = capture.time_us

    rising = np.where((y[:-1] == 0) & (y[1:] == 1))[0]
    if len(rising) < 2:
        return None

    periods_us = np.diff(t[rising])
    return 1_000_000 / np.mean(periods_us)

def run_test(freq_hz, divider, samples=4096, plot=False):
    la = PicoLogicAnalyzer(sys_clock_hz=SYSCLK)
    la.start_test_square(pin=15, frequency=freq_hz)

    time.sleep(0.1)

    capture = la.capture(
        pin_base=16,
        pin_count=1,
        samples=samples,
        divider=divider,
        trigger_pin=16,
        trigger_level=1,
        trigger_mode="edge",
    )

    measured = estimate_frequency(capture)
    sample_rate = SYSCLK / divider
    samples_per_period = sample_rate / freq_hz

    print(
        f"input={freq_hz:>9} Hz | "
        f"divider={divider:>6} | "
        f"sample_rate={sample_rate/1e6:>8.3f} MS/s | "
        f"samples/period={samples_per_period:>7.2f} | "
        f"measured={measured}"
    )

    if plot:
        la.plot(capture)
        plt.title(f"{freq_hz} Hz, {sample_rate/1e6:.2f} MS/s")
        plt.show()

    la.stop_test_square()
    return capture

tests = [
    # easy low-frequency checks
    (1_000, 150_000),   # 1 kS/s, too low for pretty 1 kHz but good slow test
    (1_000, 15_000),    # 10 kS/s, 10 samples/cycle
    (10_000, 1_500),    # 100 kS/s, 10 samples/cycle
    (10_000, 150),      # 1 MS/s, 100 samples/cycle

    # useful range
    (100_000, 150),     # 1 MS/s, 10 samples/cycle
    (500_000, 15),      # 10 MS/s, 20 samples/cycle
    (1_000_000, 15),    # 10 MS/s, 10 samples/cycle

    # push range
    (2_000_000, 5),     # 30 MS/s, 15 samples/cycle
    (5_000_000, 3),     # 50 MS/s, 10 samples/cycle
    (10_000_000, 1),    # 150 MS/s, 15 samples/cycle
]

for freq, divider in tests:
    run_test(freq, divider, plot=False)

# Plot one representative capture
run_test(100_000, 150, plot=True)