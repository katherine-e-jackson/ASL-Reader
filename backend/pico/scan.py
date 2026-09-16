import asyncio
from bleak import BleakScanner

async def scan():
    print('Scanning...')
    devices = await BleakScanner.discover(timeout=5)
    for d in devices:
        print(d.name, d.address)

asyncio.run(scan())