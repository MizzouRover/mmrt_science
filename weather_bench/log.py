"""Log the board's CSV stream to a file while echoing it to the terminal.

    python3 log.py --port /dev/cu.usbmodem101 --csv runs/uv-test.csv

Stop with Ctrl-C. Each line gets a laptop timestamp in front, so a run can be
lined up with notes like "walked outside at 14:32".
"""
import argparse
import datetime

import serial

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
ap.add_argument("--csv", required=True)
args = ap.parse_args()

with serial.Serial(args.port, 115200, timeout=1) as ser, open(args.csv, "w") as out:
    out.write("clock,ms,tempC,rh,hPa,uvi,soil\n")
    try:
        while True:
            line = ser.readline().decode(errors="replace").strip()
            # keep only data lines: they start with the board's millisecond count
            if not line or not line.split(",")[0].isdigit():
                continue
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            out.write(f"{stamp},{line}\n")
            out.flush()  # so a crash or unplug never loses the run
            print(stamp, line)
    except KeyboardInterrupt:
        pass
