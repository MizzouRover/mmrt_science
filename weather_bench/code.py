"""
What actually goes on the Feather RP2040. Copy this file and sensors.py onto the
CIRCUITPY drive when the parts arrive.

It does one job: read the sensors and print one CSV line per sample over USB.
No plotting, no filtering, no decisions. Everything else happens on the laptop,
which is what lets the GUI change without reflashing anything, and what lets the
rover computer later read this exact same stream.

To bench it without hardware, flip the import below to FakeSensors.
"""

import time

from sensors import HEADER, csv_line, RealSensors

HZ = 2.0

sensors = RealSensors()
print(HEADER)

while True:
    print(csv_line(sensors.read()))
    time.sleep(1.0 / HZ)
