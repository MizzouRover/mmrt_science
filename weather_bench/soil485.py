"""Seeed S-Soil MT-02A RS485 soil probe, read from a Feather RP2040.

Not yet run on hardware. Register map and settings come from Seeed's
S-Soil MT-02 user manual v1.0 (section 5): default slave address 1,
9600 baud, 8 data bits, no parity, 1 stop bit, Modbus RTU.

Wiring (Feather -> SparkFun SP3485 breakout BOB-10124 -> probe):
    Feather 3V   -> breakout 3-5V      Feather GND -> breakout GND
    Feather TX   -> breakout RX-I      Feather RX  -> breakout TX-O
    Feather D5   -> breakout RTS       (direction: high = talk, low = listen)
    probe yellow -> breakout A (+)     probe white -> breakout B (-)
    probe red    -> Feather USB pin (5 V from the laptop; probe needs 3.6-30 V)
    probe black  -> Feather GND        probe green (SET) -> GND or unconnected

Runs on the laptop too: `python3 soil485.py` checks the CRC against the
manual's example frame without any hardware.
"""

ADDRESS = 1
REG_TEMP = 0x0000      # INT16, value / 100 = deg C
REG_VWC = 0x0001       # INT16, value / 100 = % volumetric water content
REG_VWC_RAW = 0x0004   # INT16, raw AD count behind the VWC number


def crc16(data):
    """Modbus CRC: catches a frame garbled on the wire."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, crc >> 8])   # low byte goes first on the wire


def request(addr, start, count):
    """Function 3, 'read holding registers'."""
    frame = bytes([addr, 0x03, start >> 8, start & 0xFF, count >> 8, count & 0xFF])
    return frame + crc16(frame)


def signed16(hi, lo):
    v = hi << 8 | lo
    return v - 0x10000 if v & 0x8000 else v   # temperature can go negative


def parse(reply, count):
    """Turn a reply frame into a list of register values, or raise."""
    if len(reply) != 5 + 2 * count:
        raise ValueError("short reply: %d bytes" % len(reply))
    if crc16(reply[:-2]) != reply[-2:]:
        # Printed whole because the manual's own example reply carries a CRC
        # that standard Modbus math disagrees with; if this fires, look here.
        raise ValueError("bad CRC in " + " ".join("%02x" % b for b in reply))
    body = reply[3:-2]
    return [signed16(body[i], body[i + 1]) for i in range(0, len(body), 2)]


class SoilProbe:
    def __init__(self, uart, addr=ADDRESS):
        self.uart = uart
        self.addr = addr

    def read(self):
        """One poll: temperature, VWC and raw count. About 30 ms at 9600 baud."""
        import time
        self.uart.reset_input_buffer()
        self.uart.write(request(self.addr, REG_TEMP, 5))   # registers 0x0000-0x0004
        expect = 5 + 2 * 5
        reply = b""
        deadline = time.monotonic() + 0.5
        while len(reply) < expect and time.monotonic() < deadline:
            chunk = self.uart.read(expect - len(reply))
            if chunk:
                reply += chunk
        regs = parse(reply, 5)
        return {"soilC": regs[0] / 100, "vwc": regs[1] / 100, "raw": regs[4]}


if __name__ == "__main__":
    try:
        import board
        import busio
    except ImportError:
        # Laptop: check against the manual's worked example (section 5.4).
        assert request(1, 0, 2) == bytes.fromhex("010300000002C40B")
        reply = bytes.fromhex("010304") + bytes.fromhex("0702071D")
        reply += crc16(reply)
        print("CRC ok, request", request(1, 0, 2).hex(" "))
        print("example reply", reply.hex(" "), "->", parse(reply, 2))
    else:
        import time
        # rs485_invert=True makes D5 go HIGH while sending, which is what the
        # SP3485's driver-enable wants. CircuitPython does this timing in
        # software on the RP2040, which is fine at 9600 baud.
        uart = busio.UART(board.TX, board.RX, baudrate=9600, timeout=0.05,
                          rs485_dir=board.D5, rs485_invert=True)
        probe = SoilProbe(uart)
        while True:
            try:
                print(probe.read())
            except ValueError as e:
                print("no good reply:", e)
            time.sleep(1)
