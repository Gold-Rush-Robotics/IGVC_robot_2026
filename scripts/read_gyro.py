#!/usr/bin/env python3
"""BNO055 compass heading reader — Jetson pins 27/28 = /dev/i2c-8, addr 0x50."""

import time
import smbus2

BUS  = 1     # /dev/i2c-8  (40-pin header pins 27=SCL, 28=SDA)
ADDR = 0x6B  # BNO055 default address

OPR_MODE     = 0x3D
NDOF_MODE    = 0x0C  # full fusion (gyro + accel + mag)
EUL_HEAD_LSB = 0x1A  # 6 bytes: Heading_L H Roll_L H Pitch_L H

def u16(raw2):
    """Unsigned 16-bit (heading is 0–359.9375°)."""
    return (raw2[1] << 8) | raw2[0]

with smbus2.SMBus(BUS) as bus:
    bus.write_byte_data(ADDR, OPR_MODE, NDOF_MODE)
    time.sleep(0.7)  # NDOF calibration settle
    while True:
        d = bus.read_i2c_block_data(ADDR, EUL_HEAD_LSB, 2)
        heading = u16(d[0:2]) / 16.0  # 1 LSB = 1/16 °
        print(f"Heading: {heading:6.2f} °")
        time.sleep(0.1)
