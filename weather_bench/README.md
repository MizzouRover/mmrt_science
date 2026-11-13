# Weather module bench rig

Fake sensors today, real sensors later, same code. Nothing here needs hardware.

## Run it

    cd weather_bench
    python3 plot.py --fake        # live chart, the GUI
    python3 bench.py              # raw CSV in the terminal, the serial monitor

Other things it does:

    python3 bench.py --csv run.csv --seconds 60     save a run
    python3 plot.py --file run.csv                  replay a saved run
    python3 plot.py --port /dev/tty.usbmodem1101    real board, needs pyserial

## What is in here

- `sensors.py` - the sensor layer. `FakeSensors` invents numbers on a scripted
  60 second timeline, `RealSensors` talks to the actual boards. Same `read()`,
  same dictionary, so swapping them is one line.
- `bench.py` - prints CSV to the terminal. No dependencies.
- `plot.py` - four strip charts. Reads CSV from fake sensors, a file, or a
  serial port. Never touches a sensor itself.
- `code.py` - what goes on the Feather's CIRCUITPY drive when the parts arrive.
  Copy it and `sensors.py` across. It only prints; the laptop does the drawing.

## Soil calibration page

    python3 -m http.server 8077
    open http://localhost:8077/soil-cal.html      # in Chrome; Safari has no Web Serial

`soil-cal.html` reads the same CSV stream straight off USB, so close `log.py` /
`plot.py` first (only one program can hold the port). It shows live counts with a
settled light, logs 10 s averages tagged with water % and reinsertion number,
fits a curve, and says whether neighbouring water levels can be told apart.
Points survive a reload; the raw stream does not, so save it into `runs/`.
**Fake data** works with no board plugged in.

## Lab guide page (whole module)

    double-click start-lab.command      # starts the server on 8077, opens Chrome

`weather-lab.html` is the full acceptance lab, built on soil-cal.html: same Web
Serial stream, nine test cards (T0 link, T1 noise floor, T2 breath, T3 outdoor
step + NWS station check, T4 stairs, T5 UV, T6 sun loading, T7 STEMMA soil, T8
soak). Each card has the procedure, Mark buttons that timestamp physical steps,
live pass/fail, and its graphs. Live reference conditions come from NWS KCOU
(temp, RH, sea-level pressure) and Open-Meteo (UV index, cloud, day/night); the
reference is frozen into each test at Finish. Results persist in the browser;
save the raw stream into `runs/`. Print gives a PDF report. Built 2026-11-03,
tested on synthetic and fake data only, not yet on the board. The RS485 probe is
not in the stream, so it is not covered.

## The wire format

    ms,tempC,rh,hPa,uvi,soil
    12430,23.40,41.2,991.30,0.040,412

Fixed first, on purpose. The GUI can be rewritten without touching firmware, the
stream drops straight into a spreadsheet, and the rover computer later reads the
same lines over ROS2.

## The scripted minute

The fake timeline runs the same four checks you would do by hand on the real
parts. If the chart does not make all four obvious, the display is not finished.

| Time | What it pretends is happening |
|---|---|
| 0-15s | indoors on a desk, baseline with a slow drift |
| 15-30s | you breathe on it: humidity spikes and decays |
| 30-45s | carried outside: UV jumps, air warms and dries |
| 45-60s | soil probe moves from dry clay to damp clay |

The last one is deliberately underwhelming. The reading moves 34 counts out of a
part that can report 200 to 2000, which is the thing worth worrying about before
spending money: in dry soil, two different sites may not look different. The
soil pane says so on the chart rather than autoscaling and hiding it.

## Two things that will bite on real hardware

Both are already handled in `RealSensors`, written down here so they do not get
rediscovered the hard way.

- `board.STEMMA_I2C()` is the plug on the Feather. `board.I2C()` is the header
  pins. The whole chain hangs off the plug, so the wrong one finds nothing and
  looks exactly like dead hardware.
- The seesaw library defaults to address `0x49`. The soil sensor answers at
  `0x36`, so it has to be passed in by hand.

Addresses on the chain, checked against the library source: BME280 `0x77`,
LTR390 `0x53`, soil `0x36`. No collisions.
