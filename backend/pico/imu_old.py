from machine import I2C, Pin
from Adafruit_BNO055 import bno055
import utime, struct
import sys

# Addresses
BNO_ADDR = 0x28
OP_MODE_REG = 0x3D
ACCEL_THRESHOLD = 0.25 

vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
pos_x, pos_y, pos_z = 0.0, 0.0, 0.0
last_time = utime.ticks_ms() 


def init_bno():
    print("Waking up BNO055...")
    for attempt in range(5):
        try:
            utime.sleep_ms(200) # Small wait between attempts
            # Force CONFIG MODE (0x00)
            i2c.writeto_mem(BNO_ADDR, OP_MODE_REG, b'\x00')
            utime.sleep_ms(100)
            
            # Reset the sensor
            i2c.writeto_mem(BNO_ADDR, 0x3F, b'\x20')
            utime.sleep(1) # Mandatory 1s wait for reboot
            
            # Set to NDOF (0x0C)
            i2c.writeto_mem(BNO_ADDR, OP_MODE_REG, b'\x0C')
            utime.sleep_ms(500)
            
            print("BNO055 Initialized Successfully!")
            return True
        except OSError as e:
            print(f"Attempt {attempt+1} failed: {e}. Retrying...")
            utime.sleep(1) # Wait longer before trying again
            
    print("BNO055 Init completely failed. Check your SDA/SCL wires for loose contacts.")
    return False

def get_imu():
    global vel_x, vel_y, vel_z, pos_x, pos_y, pos_z, last_time
    try:
        # Data Collect
        acc_data = i2c.readfrom_mem(BNO_ADDR, 0x28, 6)
        ax_raw, ay_raw, az_raw = struct.unpack('<hhh', acc_data)
        ax, ay, az = ax_raw/100.0, ay_raw/100.0, az_raw/100.0

        current_time = utime.ticks_ms()
        dt = utime.ticks_diff(current_time, last_time) / 1000.0
        last_time = current_time

        # If the TOTAL movement is very small, we assume we are at rest
        # This stops the 'ghost' velocity that causes position drift
        magnitude = (ax**2 + ay**2 + az**2)**0.5
        
        if magnitude < ACCEL_THRESHOLD: 
            ax, ay, az = 0.0, 0.0, 0.0
            vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
        else:
            vel_x += ax * dt
            vel_y += ay * dt
            vel_z += az * dt

        pos_x += vel_x * dt
        pos_y += vel_y * dt
        pos_z += vel_z * dt
        
        return ax, ay, az, pos_x, pos_y, pos_z
    except:
        return None, None, None, None, None, None