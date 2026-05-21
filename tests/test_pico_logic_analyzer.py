import numpy as np

from pslab.instrument.pico_logic_analyzer import PicoLogicAnalyzer


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
        if command == "LA:READ?":
            size = str(len(self.payload)).encode("ascii")
            self.output.extend(b"#" + str(len(size)).encode("ascii") + size)
            self.output.extend(self.payload)
            self.output.extend(b"\n")
        elif command.endswith("?"):
            responses = {
                "*IDN?": b"FOSSASIA,PSLab Pico,1.0,v0.1.0\n",
                "LA:CONF:PINB?": b"16\n",
                "LA:CONF:PINC?": b"2\n",
                "LA:CONF:SAMP?": b"4\n",
                "LA:CONF:DIV?": b"1\n",
                "LA:CONF:TRIG:PIN?": b"16\n",
                "LA:CONF:TRIG:LEV?": b"1\n",
                "LA:CONF:TRIG:MODE?": b"EDGE\n",
                "TEST:SQUARE?": b"1\n",
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


def test_decode_words_two_pins():
    states = PicoLogicAnalyzer.decode_words([57], pin_count=2, samples=4)

    assert states.tolist() == [
        [True, False, True, False],
        [False, True, True, False],
    ]


def test_capture_reads_scpi_block_and_decodes_payload():
    payload = np.array([57], dtype="<u4").tobytes()
    la = PicoLogicAnalyzer(serial_instance=FakeSerial(payload))

    capture = la.capture(samples=4)

    assert capture.words.tolist() == [57]
    assert capture.states.tolist() == [
        [True, False, True, False],
        [False, True, True, False],
    ]


def test_get_xy_returns_step_plot_arrays():
    payload = np.array([57], dtype="<u4").tobytes()
    la = PicoLogicAnalyzer(serial_instance=FakeSerial(payload))
    capture = la.capture(samples=4)

    x0, y0, x1, y1 = la.get_xy(capture)

    assert len(x0) == len(y0) == 8
    assert len(x1) == len(y1) == 8
    assert y0.tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert y1.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_capture_can_configure_trigger_mode():
    payload = np.array([57], dtype="<u4").tobytes()
    fake = FakeSerial(payload)
    la = PicoLogicAnalyzer(serial_instance=fake)

    capture = la.capture(samples=4, trigger_mode="level")

    assert capture.trigger_mode == "level"
    assert b"LA:CONF:TRIG:MODE LEVEL\n" in fake.writes


def test_test_square_helpers():
    fake = FakeSerial()
    la = PicoLogicAnalyzer(serial_instance=fake)

    la.start_test_square(pin=15, frequency=1000)
    assert la.test_square_enabled()
    la.stop_test_square()

    assert b"TEST:SQUARE:CONF 15 1000\n" in fake.writes
    assert b"TEST:SQUARE OFF\n" in fake.writes
