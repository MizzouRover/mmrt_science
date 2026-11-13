"""
The sensor layer for the weather module bench rig.

Two versions of the same thing live in here. FakeSensors invents plausible
numbers so the rest of the pipeline can be built and debugged before any parts
arrive. RealSensors talks to the actual boards. Both hand back the same
dictionary with the same keys, so nothing downstream can tell them apart, and
swapping one for the other is one line at the top of bench.py or code.py.

This file is written to import cleanly on a laptop AND on the Feather running
CircuitPython, which is why the hardware imports are wrapped in a try.
"""

import math
import time

# The exact column order that goes out over the wire. Everything downstream
# keys off this list, so the header and the rows can never drift apart.
FIELDS = ["ms", "tempC", "rh", "hPa", "uvi", "soil"]
HEADER = ",".join(FIELDS)


def csv_line(reading):
    """Turn a reading dict into one line of CSV, rounded so it stays readable."""
    r = reading
    return "%d,%.2f,%.1f,%.2f,%.3f,%d" % (
        r["ms"], r["tempC"], r["rh"], r["hPa"], r["uvi"], r["soil"]
    )


class FakeSensors:
    """
    Invents readings on a scripted 60 second timeline, so a run tells a story
    instead of drawing flat lines. The phases are the same checks you would do
    by hand on the real parts, which is the point: if the plot does not make
    these four events obvious, the display is not good enough yet.

      0-15s   sitting on a desk indoors, baseline
      15-30s  you breathe on it, humidity spikes and decays
      30-45s  you carry it outside, UV jumps and the air warms and dries
      45-60s  the soil probe moves from dry clay to slightly damp clay

    That last phase is deliberately underwhelming. It is showing the thing worth
    worrying about: in dry soil the capacitive sensor only uses a narrow slice of
    its range, so two genuinely different sites can look nearly identical.
    """

    def __init__(self, loop=True, clock=None):
        # clock is swappable so a test can step time forward instantly instead of
        # waiting a real minute to see the whole scripted run.
        self._clock = clock or time.monotonic
        self._t0 = self._clock()
        self._loop = loop
        self.phase = "indoors, baseline"

    def read(self):
        since = self._clock() - self._t0
        t = (since % 60.0) if self._loop else since
        ms = int(since * 1000)

        # Baseline indoor conditions, with a slow drift so nothing is suspiciously
        # constant. Real sensors never sit perfectly still.
        temp = 22.5 + 0.4 * math.sin(t / 9.0)
        rh = 41.0 + 1.2 * math.sin(t / 7.0)
        hpa = 991.0 + 0.3 * math.sin(t / 13.0)
        uvi = 0.04
        soil = 332

        if 15.0 <= t < 30.0:
            self.phase = "breathing on it"
            # Humidity ramps hard then decays back, which is what warm breath does.
            k = t - 15.0
            rh += 42.0 * math.exp(-k / 4.5) * min(1.0, k / 0.8)
            temp += 1.6 * math.exp(-k / 5.5) * min(1.0, k / 0.8)

        elif 30.0 <= t < 45.0:
            self.phase = "carried outside"
            k = min(1.0, (t - 30.0) / 3.0)
            # UV is the headline: indoors is essentially zero, outdoors is not.
            uvi = 0.04 + 7.6 * k
            temp += 6.5 * k
            rh -= 14.0 * k
            hpa -= 0.4 * k

        elif t >= 45.0:
            self.phase = "soil: dry clay to damp clay"
            k = min(1.0, (t - 45.0) / 2.0)
            # 332 to 366. A real change, and a depressingly small one.
            soil = int(332 + 34 * k)
            uvi = 0.04

        else:
            self.phase = "indoors, baseline"

        return {
            "ms": ms,
            "tempC": temp,
            "rh": max(0.0, min(100.0, rh)),
            "hPa": hpa,
            "uvi": max(0.0, uvi),
            "soil": soil,
        }


class RealSensors:
    """
    The actual parts. Not used until the box arrives, but written now so the
    shape of read() is locked in and the fake version cannot drift away from it.

    Two things here are easy to get wrong and hard to debug:
      - board.STEMMA_I2C() is the plug on the end of the Feather. board.I2C() is
        the header pins. The whole chain hangs off the plug, so the wrong one
        finds nothing and looks exactly like dead hardware.
      - the seesaw library defaults to address 0x49, but the soil sensor answers
        at 0x36, so it has to be passed in by hand.
    """

    def __init__(self):
        import board
        import adafruit_bme280.basic as bme280_mod
        import adafruit_ltr390
        from adafruit_seesaw.seesaw import Seesaw

        i2c = board.STEMMA_I2C()
        self._t0 = time.monotonic()
        self.bme = bme280_mod.Adafruit_BME280_I2C(i2c, address=0x77)
        self.uv = adafruit_ltr390.LTR390(i2c, address=0x53)
        # The STEMMA soil sensor is optional now that the RS485 probe is taking
        # its place: unplugged, the soil column reads -1 instead of crashing.
        try:
            self.soil = Seesaw(i2c, addr=0x36)
        except ValueError:
            self.soil = None
        self.phase = "live"

        # 18x gain, 16-bit: about 144 counts per UVI (0.007 UVI per count), and
        # the 65535-count ceiling is ~450 UVI, so full sun cannot peg it. At 1x
        # one count was 0.125 UVI, too coarse to trust (2026-10-28). The
        # library's .uvi rescales for gain on its own, so nothing else changes.
        self.uv.gain = adafruit_ltr390.Gain.GAIN_18X
        self.uv.resolution = adafruit_ltr390.Resolution.RESOLUTION_16BIT

    def read(self):
        return {
            "ms": int((time.monotonic() - self._t0) * 1000),
            "tempC": self.bme.temperature,
            "rh": self.bme.relative_humidity,
            "hPa": self.bme.pressure,
            "uvi": self.uv.uvi,
            "soil": self.soil.moisture_read() if self.soil else -1,
        }
