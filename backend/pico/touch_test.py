from machine import Pin, I2C

import time

print("start")
i2c = I2C(0, sda=Pin(4), scl=Pin(5))
print(i2c)
print("Connected touch!")

mpr = MPR121(i2c)

while True:
    if mpr.is_touched(0):
        print("touched!")

    time.sleep(0.1)