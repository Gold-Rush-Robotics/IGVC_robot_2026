from smbus2 import SMBus

# Initialize I2C bus 1 (standard for Raspberry Pi)
with SMBus(1) as bus:
    # Write a byte to a register (e.g., config register)
    bus.write_byte_data(0x6B, 0x3D, 0x0C)

    # Read a byte from a register (e.g., temperature reading)
    data = bus.read_byte_data(0x6B, 0x1A, 2)
    print(f"Sensor value: {data}")
