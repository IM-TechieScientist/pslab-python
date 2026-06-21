# PSLab Pico Python Tools

This is a fork that is primarily intended for development and testing of the new RP2350 based PSLab.

It talks to the board over USB CDC using SCPI commands and provides Python helpers for the currently
implemented instruments:

- Pico logic analyser
- Pico oscilloscope using the internal ADC
- Built-in test square-wave output
- Optional ESP SPI / Wi-Fi data transport for streamed capture preview
- Development waveform workbench GUI


## Install

Use a virtual environment from the repository root:

```bash
cd /home/santosh/pslab_pico/pslab-python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install PyQt5
```

For a minimal non-GUI install, `python3 -m pip install -e .` is enough.
`PyQt5` is needed only for the waveform workbench.

On Linux, make sure your user can access USB CDC serial devices. Usually this
means joining the `dialout` group:

```bash
sudo usermod -aG dialout "$USER"
```

## Quick Logic Analyser Test

Connect the Pico running the PSLab Pico firmware over USB.

```python
from pslab.instrument.pico_logic_analyzer import PicoLogicAnalyzer

la = PicoLogicAnalyzer()
print(la.identify())

la.start_test_square(pin=15, frequency=1000)
capture = la.capture(
    pin_base=16,
    pin_count=1,
    samples=1024,
    divider=1500,
    trigger_pin=16,
    trigger_level=True,
    trigger_mode="edge",
)

print(capture.sample_rate)
print(capture.states.shape)
la.stop_test_square()
la.disconnect()
```

For this test, wire `GP15 -> GP16` and connect ground as usual.

## Quick Oscilloscope Test

The current Pico oscilloscope uses the RP2350 internal ADC. Channels `0..3`
map to GPIOs `26..29`.

```python
from pslab.instrument.pico_oscilloscope import PicoOscilloscope

scope = PicoOscilloscope()
capture = scope.capture(
    channel=0,
    sample_rate=100_000,
    samples=1024,
    trigger_mode="OFF",
)

print(capture.gpio)
print(capture.volts[:10])
scope.disconnect()
```

## SCPI Access

The Python classes expose direct SCPI helpers:

```python
from pslab.instrument.pico_logic_analyzer import PicoLogicAnalyzer

dev = PicoLogicAnalyzer()
print(dev.query("*IDN?"))
dev.command("COMM:TRAN USB")
print(dev.query("COMM:TRAN?"))
print(dev.query("SYST:ERR?"))
dev.disconnect()
```

Use `command()` for SCPI commands without a response and `query()` for commands
ending in `?`.

## Waveform Workbench GUI

The main development GUI is:

```bash
cd /home/santosh/pslab_pico/pslab-python
source .venv/bin/activate
python3 tools/pico_waveform_workbench.py
```

The workbench supports:

- USB CDC SCPI connection selection
- Logic analyser capture and repeated capture
- Pico ADC oscilloscope capture
- Built-in test square-wave control
- Trigger mode, trigger level, samples, divider/rate, channel settings
- Zoom, scroll, fit, fullscreen, channel visibility
- Latest-capture mode for waveform correctness
- Rolling timeline mode for preview-style display
- UDP receiver for ESP SPI / Wi-Fi transport frames
- Transport selection: `USB`, `WIFI`, `AUTO`
- Wi-Fi transport counters and stream diagnostics

## ESP SPI / Wi-Fi Transport

SCPI commands always go over USB CDC. Stream data can optionally be sent by the
firmware over the Pico-to-ESP SPI bridge and forwarded by the ESP over UDP.

In Python/GUI terms:

1. Start the ESP bridge receiver path.
2. Start the GUI Wi-Fi receiver.
3. Set transport to `WIFI` or `AUTO`.
4. Start LA or DSO repeated streaming.

This transport is useful for live preview. It is not a lossless high-rate
continuous acquisition path; dropped buffers or skipped display frames should be
expected at higher sample rates.