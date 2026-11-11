"""Hardware-free checks for the weather bench. Run from the repo root:

    python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "weather_bench"))

import sensors  # noqa: E402
import soil485  # noqa: E402


class StepClock:
    """A clock the test moves by hand, so the 60 s fake run takes no real time."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class CsvFormat(unittest.TestCase):
    def test_header_matches_fields(self):
        self.assertEqual(sensors.HEADER, "ms,tempC,rh,hPa,uvi,soil")

    def test_line_has_one_value_per_field(self):
        clock = StepClock()
        line = sensors.csv_line(sensors.FakeSensors(clock=clock).read())
        self.assertEqual(len(line.split(",")), len(sensors.FIELDS))


class FakeTimeline(unittest.TestCase):
    def reading_at(self, seconds):
        clock = StepClock()
        fake = sensors.FakeSensors(clock=clock)
        clock.now = seconds
        return fake.read()

    def test_breath_spikes_humidity(self):
        self.assertGreater(self.reading_at(17)["rh"], self.reading_at(5)["rh"] + 15)

    def test_outdoors_raises_uv(self):
        self.assertGreater(self.reading_at(40)["uvi"], 5)

    def test_soil_moves_but_only_a_little(self):
        dry, damp = self.reading_at(5)["soil"], self.reading_at(55)["soil"]
        self.assertGreater(damp, dry)
        self.assertLess(damp - dry, 100)

    def test_humidity_stays_in_range(self):
        for s in range(0, 60):
            self.assertTrue(0.0 <= self.reading_at(s)["rh"] <= 100.0)


class Rs485Soil(unittest.TestCase):
    def test_request_matches_manual_example(self):
        # S-Soil MT-02 manual section 5.4: read 2 registers from address 1
        self.assertEqual(soil485.request(1, 0, 2), bytes.fromhex("010300000002C40B"))

    def test_parse_round_trip(self):
        body = bytes.fromhex("010304") + bytes.fromhex("0702071D")
        self.assertEqual(soil485.parse(body + soil485.crc16(body), 2), [1794, 1821])

    def test_bad_crc_is_rejected(self):
        body = bytes.fromhex("010304") + bytes.fromhex("0702071D")
        with self.assertRaises(ValueError):
            soil485.parse(body + b"\x00\x00", 2)

    def test_negative_temperature(self):
        self.assertEqual(soil485.signed16(0xFF, 0x38), -200)


if __name__ == "__main__":
    unittest.main()
