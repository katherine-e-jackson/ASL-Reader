import network
import utime

# Initialize the Wi-Fi chip
wlan = network.WLAN(network.STA_IF)
wlan.active(True)

# Replace with your actual credentials
ssid = 'Your_Network_Name'
password = 'Your_Password'

if not wlan.isconnected():
    print('Connecting to network...')
    wlan.connect(ssid, password)
    
    # Wait for connection (timeout after 10 seconds)
    timeout = 10
    while not wlan.isconnected() and timeout > 0:
        print('.')
        utime.sleep(1)
        timeout -= 1

if wlan.isconnected():
    print('Connected!')
    status = wlan.ifconfig()
    print('IP Address: ' + status[0])
    print('Subnet Mask: ' + status[1])
    print('Gateway: ' + status[2])
    print('DNS: ' + status[3])
else:
    print('Connection failed. Check your SSID and Password.')