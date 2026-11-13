"""
Prints readings as CSV, one line per sample. This is the serial-monitor stand-in:
no graphics, no dependencies, just the raw stream so you can see what the board
would actually be sending.

    python3 bench.py                 fake sensors, prints to the terminal
    python3 bench.py --csv run.csv   also writes every line to a file
    python3 bench.py --hz 10         faster sampling
"""

import argparse
import sys
import time

from sensors import HEADER, csv_line, FakeSensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="also write the stream to this file")
    ap.add_argument("--hz", type=float, default=2.0, help="samples per second")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds, 0 runs forever")
    args = ap.parse_args()

    sensors = FakeSensors()
    out = open(args.csv, "w") if args.csv else None
    period = 1.0 / args.hz
    started = time.monotonic()

    print(HEADER)
    if out:
        out.write(HEADER + "\n")

    try:
        while True:
            line = csv_line(sensors.read())
            print(line)
            sys.stdout.flush()
            if out:
                out.write(line + "\n")
                out.flush()
            if args.seconds and (time.monotonic() - started) >= args.seconds:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()


if __name__ == "__main__":
    main()
