from machine import Pin, SoftI2C, ADC
from ads1x15 import ADS1015
from bno055 import BNO055
from mpr121 import MPR121
from ble_simple_peripheral import BLESimplePeripheral  # ADD
import bluetooth                                        # ADD
import machine
import utime

# --- BLE Setup ---
ble = bluetooth.BLE()
sp = BLESimplePeripheral(ble, name="ASLG")

# Starting Calibration values for flex sensors
# 0 -> thumb
min_0 = 33000
max_0 = 43000
# 1 -> index 
min_1 = 28000
max_1 = 43000
# 2 -> middle
min_2 = 28000
max_2 = 38000
# 3 -> ring
min_3 = 700
max_3 = 1000
# 4 -> pinky
min_4 = 700
max_4 = 1000

# Teadoras Cal
# min_0 = 35000
# max_0 = 43000
# # 1 -> index 
# min_1 = 13000
# max_1 = 43000
# # 2 -> middle
# min_2 = 30000
# max_2 = 38000
# # 3 -> ring
# min_3 = 800
# max_3 = 1000
# # 4 -> pinky
# min_4 = 800
# max_4 = 1000


vals = []
perc = []

# For calibration 
imu_cal = False
flex_cal = False

# Assign each flex sensor to pins 
thumb_flex = ADC(Pin(28))   # Pin 31 ADC0
index_flex = ADC(Pin(27))   # Pin 32 ADC1
middle_flex = ADC(Pin(26))  # Pin 34 ADC2

# I2C ADS @ Pin 9 and 10 
i2c_ads = SoftI2C(sda=Pin(6), scl=Pin(7))

# I2C IMU @ Pin 21 and 22
i2c_imu = SoftI2C(sda=Pin(12), scl=Pin(13))

# I2C MPR @ Pin 4 and 5
i2c_mpr = SoftI2C(sda=Pin(4), scl=Pin(5))

# Instantiate IMU
imu = BNO055(i2c_imu)

# ADS at address 48
ads = ADS1015(i2c_ads, address= 0x48, gain=1)

# MPR 
mpr = MPR121(i2c_mpr) 

# Calibrate flex sensors for 5 seconds 
# takes in a flex pin to read
# returns -> min and max reading
def calibrate_flex():
    global min_0, min_1, min_2, min_3, min_4, max_0, max_1, max_2, max_3, max_4

    print("CALIBRATION START")
    
    start_time = utime.ticks_ms()

    min_read = [min_0, min_1, min_2, min_3, min_4]
    max_read = [max_0, max_1, max_2, max_3, max_4]

    # while utime.ticks_diff(utime.ticks_ms(), start_time) < 5000:
    #     thumb_raw = thumb_flex.read_u16()
    #     index_raw = index_flex.read_u16()
    #     middle_raw = middle_flex.read_u16()
    #     ring_raw = ads.read(rate=4, channel1=1)  # Pin A1 on ADS from I2C
    #     pinky_raw = ads.read(rate=4, channel1=0) # Pin A0 on ADS from I2C

    #     flex_val = [thumb_raw, index_raw, middle_raw, ring_raw, pinky_raw]
    #     for i in range(5):

    #         if flex_val[i] < min_read[i]: min_read[i] = flex_val[i]
    #         if flex_val[i] > max_read[i]: max_read[i] = flex_val[i]

    #         utime.sleep_ms(10)
        
    #     min_0, min_1, min_2, min_3, min_4 = min_read
    #     max_0, max_1, max_2, max_3, max_4 = max_read

    print("CALIBRATION DONE")
    print("Min:", min_read, "Max:", max_read)
    return min_read, max_read

# Calculates the percentage straight
def percent_straight(val, finger):
    if (finger == 0):
        MIN = min_0
        MAX = max_0
    elif (finger == 1):
        MIN = min_1
        MAX = max_1
    elif (finger == 2):
        MIN = min_2
        MAX = max_2
    elif (finger == 3):
        MIN = min_3
        MAX = max_3
    elif (finger == 4):
        MIN = min_4
        MAX = max_4
    else:
        print( "invalid finger given")
        return 

    p = (val - MIN) / (MAX - MIN) * 100
    return max(0, min(100, p))
  
# Load in accleration and rotation data from IMU
def get_imu():
    global imu

    imu_data = [
        imu.accel(),
        imu.euler()
        ]

    return imu_data

# Flex sensors values
def get_flex():
    try:
        thumb_raw = thumb_flex.read_u16()
        index_raw = index_flex.read_u16()
        middle_raw = middle_flex.read_u16()

        ring_raw = ads.read(rate=4, channel1=1)  # Pin A1 on ADS from I2C
        pinky_raw = ads.read(rate=4, channel1=0) # Pin A0 on ADS from I2C

        vals = [thumb_raw, index_raw, middle_raw, ring_raw, pinky_raw]

        # Testing Feature: Prints raw values for calibration
        # print("RAW     -> T: {:5d} | I: {:5d} | M: {:5d} | R: {:4d} | P: {:4d}".format(*vals))

        # Calculate the percentage bent for each finger
        perc = [percent_straight(vals[i], i) for i in range(5)]
        return perc
    
    except OSError:
        return None 

'''
thumb, index, middle, ring, pinky - straight percentage
accel_x, accel_y, accel_z -  acceleration
heading - up down nose (x-axis motion)
rolling - side to side (y-axis motion)
pitch - nose shake (z-axis)
touch_thumb, touch_point - touch sensors     
'''
def get_data():

    flex_data = get_flex()
    imu_data = get_imu()

    # Thumb Touch
    if mpr.is_touched(0):
        touch_thumb = 1
    else: 
        touch_thumb = 0

    # Pointer Touch
    if mpr.is_touched(1):
        touch_point = 1
    else: 
        touch_point  = 0

    # Pointer Touch
    if mpr.is_touched(2):
        middle_point = 1
    else: 
        middle_point  = 0

    # If the data is valid 
    if imu_data and flex_data:
        thumb, index, middle, ring, pinky = flex_data

        accel, euler = imu_data
        accel_x, accel_y, accel_z = accel
        heading, rolling, pitch = euler

        line = "{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{}".format(
        thumb, index, middle, ring, pinky,  # from flex_read.get_flex()
        accel_x, accel_y, accel_z,          # from imu_read.get_imu()
        heading, rolling, pitch,            # from imu_read.get_imu()
        touch_thumb, touch_point, middle_point            # from mpr.is_touched()
        )

        # line_pretty = "FLEX: \n Thumb: {:.1f}, Index: {:.1f}, Middle: {:.1f}, Ring: {:.1f}, Pinky: {:.1f} \nACCELERATION: \n x: {:.2f}, y: {:.2f}, z: {:.2f} \n" \
        # "EULER: \n heading: {:.2f}, rolling: {:.2f}, pitch: {:.2f} \nTOUCH SENSORS: Thumb: {}, Index: {}, Middle: {}".format(
        # thumb, index, middle, ring, pinky,  # from flex_read.get_flex()
        # accel_x, accel_y, accel_z,          # from imu_read.get_imu()
        # heading, rolling, pitch,            # from imu_read.get_imu()
        # bool(touch_thumb), bool(touch_point), bool(middle_point)            # from mpr.is_touched()
        # )
        # print(line_pretty)
        # print("-----------------------------------")

        # Only send if BLE is connected
        if sp.is_connected():
            sp.send(line)


        # Testing Feature classify bentness
        # flex_class = ["straight" if p > 50 else "bent" for p in flex_data]

        return thumb, index, middle, ring, pinky, accel_x, accel_y, accel_z, heading, rolling, pitch, touch_thumb, touch_point, middle_point

    else:
        print("theres an issue...")

    return None

utime.sleep_ms(500)
while True:
    # Check IMU cal status but don't block
    if not imu_cal:
        imu_cal = imu.calibrated()

    # Calibrate flex once
    if not flex_cal:
        calibrate_flex()
        flex_cal = True

    # Always get and send data, even if IMU not fully calibrated
    get_data()
    utime.sleep_ms(50)