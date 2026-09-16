import bluetooth
from micropython import const

_IRQ_CENTRAL_CONNECT    = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_FLAG_NOTIFY = const(0x0010)
_FLAG_READ   = const(0x0002)

_SENSOR_SERVICE_UUID = bluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
_SENSOR_CHAR_UUID    = bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")

_SENSOR_SERVICE = (
    _SENSOR_SERVICE_UUID,
    ((_SENSOR_CHAR_UUID, _FLAG_READ | _FLAG_NOTIFY),),
)

ble = bluetooth.BLE()
ble.active(True)

((handle,),) = ble.gatts_register_services((_SENSOR_SERVICE,))

name = "PicoSensor"
name_bytes = name.encode("utf-8")
adv_data = bytes([len(name_bytes) + 1, 0x09]) + name_bytes
ble.gap_advertise(100_000, adv_data)

print("Advertising as PicoSensor...")

import time
while True:
    time.sleep(1)
    print("still advertising...")