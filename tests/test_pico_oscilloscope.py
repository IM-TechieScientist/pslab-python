import numpy as np

from pslab.instrument.pico_oscilloscope import PicoOscilloscope


class FakeSerial:
    def __init__(self, payload=b""):
        self.payload = payload
        self.output = bytearray()
        self.writes = []
        self.timeout = 1
        self.is_open = True

    def write(self, data):
        self.writes.append(data)
        command = data.strip().decode("ascii")
        if command == "DSO:READ?":
            size = str(len(self.payload)).encode("ascii")
            self.output.extend(b"#" + str(len(size)).encode("ascii") + size)
            self.output.extend(self.payload)
            self.output.extend(b"\n")
        elif command.endswith("?"):
            responses = {
                "*IDN?": b"FOSSASIA,PSLab Pico,1.0,v0.1.0\n",
                "DSO:CONF:CHAN?": b"0\n",
                "DSO:CONF:GPIO?": b"26\n",
                "DSO:CONF:RATE?": b"100000\n",
                "DSO:CONF:SAMP?": b"4\n",
                "DSO:CONF:TRIG:LEV?": b"2048\n",
                "DSO:CONF:TRIG:MODE?": b"EDGE\n",
                "DSO:CONF:TRIG:SLOP?": b"RISE\n",
                "DSO:STREAM:STAT?": b"0,3,0\n",
                "SYST:ERR?": b"0,\"No error\"\n",
            }
            self.output.extend(responses[command])
        else:
            self.output.extend(b"OK\n")
        return len(data)

    def read(self, size=1):
        data = self.output[:size]
        del self.output[:size]
        return bytes(data)

    def readline(self):
        try:
            end = self.output.index(ord("\n")) + 1
        except ValueError:
            end = len(self.output)
        return self.read(end)

    def close(self):
        self.is_open = False


def test_scope_capture_reads_uint16_samples():
    payload = np.array([0, 2048, 4095, 1024], dtype="<u2").tobytes()
    scope = PicoOscilloscope(serial_instance=FakeSerial(payload))

    capture = scope.capture(samples=4)

    assert capture.raw.tolist() == [0, 2048, 4095, 1024]
    assert capture.samples == 4
    assert capture.volts[0] == 0
    assert capture.volts[2] == 3.3


def test_scope_configures_trigger_options():
    payload = np.array([1, 2], dtype="<u2").tobytes()
    fake = FakeSerial(payload)
    scope = PicoOscilloscope(serial_instance=fake)

    capture = scope.capture(
        channel=1,
        sample_rate=200000,
        samples=2,
        trigger_level=1000,
        trigger_mode="edge",
        trigger_slope="falling",
    )

    assert capture.channel == 1
    assert capture.gpio == 26
    assert capture.trigger_mode == "EDGE"
    assert capture.trigger_slope == "FALL"
    assert b"DSO:CONF:CHAN 1\n" in fake.writes
    assert b"DSO:CONF:RATE 200000\n" in fake.writes
    assert b"DSO:CONF:SAMP 2\n" in fake.writes
    assert b"DSO:CONF:TRIG:LEV 1000\n" in fake.writes
    assert b"DSO:CONF:TRIG:MODE EDGE\n" in fake.writes
    assert b"DSO:CONF:TRIG:SLOP FALL\n" in fake.writes


def test_scope_stream_status():
    scope = PicoOscilloscope(serial_instance=FakeSerial())

    status = scope.stream_status()

    assert not status.enabled
    assert status.sequence == 3
    assert status.overruns == 0
