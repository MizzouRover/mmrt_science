"""
The basic GUI. Four stacked strip charts, one per reading, on a rolling window.

It never touches a sensor. It reads CSV lines, from whichever of three places
you point it at, which is the whole reason the wire format was fixed first:

    python3 plot.py --fake                     invented data, no hardware needed
    python3 plot.py --file run.csv             replay a saved run
    python3 plot.py --port /dev/tty.usbmodem*  the real Feather (needs pyserial)

The soil pane carries a note about the sensor's full range. That is deliberate.
The reading moves, but it moves inside a very small slice of what the part can
report, and a chart that autoscales will hide that from you.
"""

import argparse
import collections
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sensors import FIELDS, FakeSensors

WINDOW_S = 60.0          # seconds of history kept on screen
SOIL_FULL = (200, 2000)  # what the datasheet says the part can report

PANES = [
    ("tempC", "air temp", "deg C", "#d85a30"),
    ("rh", "humidity", "% RH", "#378add"),
    ("uvi", "UV index", "UVI", "#7f77dd"),
    ("soil", "soil moisture", "counts", "#1d9e75"),
]


def parse_line(line, fields):
    """Turn one CSV row into a dict, or return None if it is a header or junk."""
    parts = line.strip().split(",")
    if len(parts) != len(fields):
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None          # this is the header row
    return dict(zip(fields, vals))


class Source:
    """Wraps the three places readings can come from behind one next() call."""

    def __init__(self, args, clock=None):
        self.kind = "fake"
        self.phase = ""
        if args.port:
            import serial
            self.kind = "serial"
            self.ser = serial.Serial(args.port, 115200, timeout=0.05)
        elif args.file:
            self.kind = "file"
            self.rows = [r for r in (parse_line(l, FIELDS) for l in open(args.file)) if r]
            self.i = 0
        else:
            self.sensors = FakeSensors(clock=clock)

    def next(self):
        """Return the newest reading, or None if nothing is available yet."""
        if self.kind == "fake":
            self.phase = self.sensors.phase
            return self.sensors.read()
        if self.kind == "file":
            if self.i >= len(self.rows):
                self.i = 0
            row = self.rows[self.i]
            self.i += 1
            return row
        line = self.ser.readline().decode("utf-8", "replace")
        return parse_line(line, FIELDS) if line else None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fake", action="store_true", help="invented readings, the default")
    g.add_argument("--file", help="replay a CSV file")
    g.add_argument("--port", help="serial port of a real board")
    ap.add_argument("--save", help="render one frame to a PNG and exit, for checking")
    args = ap.parse_args()

    stepper = None
    if args.save:
        matplotlib.use("Agg")
        # Step the scripted minute forward 0.25 s per frame so one saved image
        # shows the whole run instead of a quarter second of it.
        state = {"t": 0.0}

        def stepper():
            state["t"] += 0.25
            return state["t"]

    src = Source(args, clock=stepper)

    t = collections.deque()
    series = {k: collections.deque() for k, _, _, _ in PANES}
    t0 = time.monotonic()

    fig, axes = plt.subplots(len(PANES), 1, figsize=(9, 8), sharex=True)
    fig.canvas.manager.set_window_title("Weather module bench")
    lines = {}
    for ax, (key, title, unit, color) in zip(axes, PANES):
        (ln,) = ax.plot([], [], color=color, linewidth=1.6)
        lines[key] = ln
        ax.set_ylabel("%s\n%s" % (title, unit), fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("seconds", fontsize=9)

    soil_ax = axes[-1]
    soil_note = soil_ax.text(
        0.012, 0.93, "", transform=soil_ax.transAxes, ha="left", va="top",
        fontsize=8, color="#0f6e56",
    )
    banner = fig.suptitle("", fontsize=11)

    def update(_frame):
        # One sample per frame when faking or replaying. On a real port, take
        # whatever has piled up in the buffer so the chart cannot fall behind.
        budget = 40 if src.kind == "serial" else 1
        for _ in range(budget):
            r = src.next()
            if r is None:
                break
            # The board's own millisecond counter is the time base, not the
            # laptop's clock. Same column the rover computer will read later.
            t.append(r["ms"] / 1000.0)
            for key, _, _, _ in PANES:
                series[key].append(r[key])

        if not t:
            return list(lines.values())

        now = t[-1]
        while len(t) > 1 and (now - t[0]) > WINDOW_S:   # drop what fell off the left
            t.popleft()
            for key, _, _, _ in PANES:
                series[key].popleft()

        for ax, (key, _, _, _) in zip(axes, PANES):
            lines[key].set_data(list(t), list(series[key]))
            ax.set_xlim(max(0.0, now - WINDOW_S), max(WINDOW_S, now))
            lo, hi = min(series[key]), max(series[key])
            pad = (hi - lo) * 0.15 or 0.5
            ax.set_ylim(lo - pad, hi + pad)

        lo, hi = min(series["soil"]), max(series["soil"])
        soil_note.set_text(
            "this window spans %d counts of the %d to %d the part can report"
            % (hi - lo, SOIL_FULL[0], SOIL_FULL[1])
        )
        label = src.phase if src.kind == "fake" else src.kind
        banner.set_text("fake sensors  |  %s" % label if src.kind == "fake" else label)
        return list(lines.values())

    anim = FuncAnimation(fig, update, interval=120, blit=False, cache_frame_data=False)

    if args.save:
        for _ in range(244):                   # 61 seconds at 0.25 s a frame
            update(0)
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print("wrote", args.save)
        return
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
