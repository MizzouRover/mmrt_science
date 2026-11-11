# mmrt_science

Platform-level code for the science module on MMRTs 2026 mars rover.
Nothing here is ROS. It runs on a laptop, or on the Feather board in the weather module.

| Folder | What it is | Status |
| --- | --- | --- |
| `weather_bench/` | Weather module bench rig: air temp, humidity, pressure, UV and soil moisture. Fake sensors today, real sensors behind the same interface. Includes live charts, a browser lab page and the RS-485 soil probe reader. | Runs on fake data. STEMMA sensors and the RS-485 probe not yet tested on hardware. |
| `panorama/` | Stitches overlapping camera photos into one wide panorama with OpenCV. | Tested on the Nov 6 Navcam set and the Utah test frames. |
| `tests/` | Checks that need no hardware. | `python3 -m unittest discover -s tests -v` |

## Quick start

Weather bench, no hardware needed:

    cd weather_bench
    pip install -r requirements.txt
    python3 bench.py --seconds 5      # raw CSV in the terminal
    python3 plot.py --fake            # live strip charts

Panorama (put photos in `panorama/captures/`, or pass `--input some_folder`):

    cd panorama
    uv venv .venv && uv pip install -r requirements.txt
    python panorama.py --show

Each folder has more detail in its own README or file header.
Photos, virtual environments and recorded bench runs are git-ignored on purpose.
