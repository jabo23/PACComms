"""
Receives IR camera blob data over Bluetooth from ESP32,
solves PnP with OpenCV to determine gun pose in 3D space.

    pip install opencv-python-headless numpy pyserial
    pip install pyautogui
    py -m pip install pyautogui

    python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__) to verify
    python3 -c "import serial; print(serial.__version__) to verify

    Pair the ESP32 to the Pi first, then it connects to /dev/rfcomm0

    x1,y1,x2,y2,x3,y3,x4,y4
    - 4 IR blob centroids in pixel coords (0-1023 range from PixArt)
    - If a blob is not visible, its coords are sent as -1,-1
"""
"""
TO TEST THE FILE
run python3 TestData.py
OR
use socat and open three SSH sessions to make a virtual port (YOU WILL NEED TO CHANGE line 41 "SERIAL_PORT   = "/dev/rfcomm0" to SERIAL_PORT   = "/tmp/fake" or whatever you want to call it)
then on the other session run Algo.py (python3 Algo.py)
then on the last session manually send dummy data ex: echo "100,100 100,100 100,100 100,100" > /tmp/fake

"""
import cv2
import numpy as np
import sys
import logging
import pyautogui
import asyncio
from bleak import *
import time
import threading
import queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("controller")

# BLE comms with controller
DEVICE_NAME = "PACController"
DATA_CHARACTERISTIC_UUID = 'ffeddbc9-b7a5-9381-7f6d-5b4937251301'

# PixArt sensor resolution 
# https://pmc.ncbi.nlm.nih.gov/articles/PMC7218719/ this better be right
SENSOR_W = 1023
SENSOR_H = 767

pyautogui.FAILSAFE = False # prevents crash at corners
pyautogui.PAUSE = 0 # because pyautogui is SLOOOOWWWWWWW by default for some reason

SCREEN_W, SCREEN_H = pyautogui.size() #gets screen resolution

# Physical IR LED positions in the WORLD frame (metres, origin = top-left LED).
LED_SPACING_X = 0.24   # metres between left and right LEDs
LED_SPACING_Y = 0.12 # metres between top and bottom LEDs
# These are estimates based on our screen size if it is rotated
# update for accuracy on actual DEMO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

WORLD_POINTS = np.array([
    [0.0,          0.0,          0.0],   # Top-left
    [LED_SPACING_X, 0.0,          0.0],   # Top-right
    [LED_SPACING_X, LED_SPACING_Y, 0.0],  # Bottom-right
    [0.0,          LED_SPACING_Y, 0.0],   # Bottom-left
], dtype=np.float64)

# Camera intrinsics 
# Run camera_calibrate.py (or OpenCV calibration) to get exact values.
FOCAL_LENGTH_PX = 600.0
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_PX, 0.0,              SENSOR_W / 2],
    [0.0,             FOCAL_LENGTH_PX,  SENSOR_H / 2],
    [0.0,             0.0,              1.0          ]
], dtype=np.float64)

# Distortion coefficients - zero until calibration is actually done
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def parse_packet(data: bytearray):
    """
    Parse a byte array of 9 comma-separated hex values: button,x1,y1,x2,y2,x3,y3,x4,y4
    button is 0x00 or 0x01
    coords are hex integers representing pixel positions
    Returns (blobs, button) or (None, 0) if malformed.
    """
    try:

        samples = [data[offset:(offset+20)] for offset in range(0, len(data), 20)]

        ret = []

        for sample in samples:
            values = [int(b) for b in sample]
            if len(values) != 20:
                return None
            button = values[0]
            delay = values[2]
            coords = values[4:]
            blobs = []
            for i in range(0, 16, 4):
                x = coords[i + 0] + coords[i + 1] * 256
                y = coords[i + 2] + coords[i + 3] * 256
                if x < 2000 and y < 2000:
                    blobs.append((x, y))
            ret.append((blobs, button, delay))

        return ret
    except ValueError:
        return None


def solveController(blobs):
    """
    Attempt to solve PnP given a list of (x, y) image-space blob coords.

    The PixArt camera reports blobs in order of brightness, not spatial order.
    For a reliable solve we need exactly 4 blobs and sort them into a
    consistent spatial order (top-left, top-right, bottom-right, bottom-left).

    Returns (rvec, tvec, screen_xy) or (None, None, None) on failure.
      - rvec: rotation vector (3x1)
      - tvec: translation vector (3x1, metres from camera to LED origin)
      - screen_xy: estimated aim point as (x, y) fraction of screen (0.0-1.0)
    """
    if len(blobs) < 3: #immediate FAIL on less than 3 blobs (pnp req 3 or more to work)
        # log.debug("Not enough blobs: %d", len(blobs))
        return None, None, None

    if len(blobs) == 4:
        pts = sort_quad(blobs)
        world_pts = WORLD_POINTS
    elif len(blobs) == 3:
        # in the event one LED is obfuscated for whatever reason
        pts = np.array(blobs[:3], dtype=np.float64)
        world_pts = WORLD_POINTS[:3]
    else:
        pts = np.array(blobs[:4], dtype=np.float64)
        world_pts = WORLD_POINTS

    image_pts = np.array(pts, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        world_pts,
        image_pts,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_SQPNP if len(world_pts) == 3 else cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None, None, None

    centre_world = np.array([[[LED_SPACING_X / 2, LED_SPACING_Y / 2, 0.0]]], dtype=np.float64)
    centre_img, _ = cv2.projectPoints(centre_world, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    cx = float(centre_img[0][0][0]) / SENSOR_W
    cy = float(centre_img[0][0][1]) / SENSOR_H

    return rvec, tvec, (cx, cy)


def sort_quad(blobs):
    """
    Sort 4 blobs into (top-left, top-right, bottom-right, bottom-left) order
    using the same approach as OpenCV's findHomography helpers.
    """
    pts = np.array(blobs, dtype=np.float64)
    # Sort by Y first (top vs bottom)
    pts = pts[np.argsort(pts[:, 1])]
    top    = pts[:2][np.argsort(pts[:2, 0])]    # sort top two by X
    bottom = pts[2:][np.argsort(pts[2:, 0])]    # sort bottom two by X
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


ALPHA = 0.7

def on_pose_solved(screen_xy, oldpx, oldpy):
    x_frac, y_frac = screen_xy
    x_frac = max(0.0, min(1.0, x_frac))
    y_frac = max(0.0, min(1.0, y_frac))
    px = SCREEN_W - int(x_frac * SCREEN_W)
    py = int(y_frac * SCREEN_H)

    if oldpx < 0 or oldpy < 0:
        oldpx = px
        oldpy = py
    px = int(px * ALPHA + oldpx * (1 - ALPHA))
    py = int(py * ALPHA + oldpy * (1 - ALPHA))

    pyautogui.moveTo(px, py, _pause=False)

    return px, py

def onButton(button: int):
    if button == 1:
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()

def on_no_lock():
    pass

class Connection:
    def __init__(self):
        self.status: bool = False

    def set_status(self, val: bool):
        self.status = val

    def get_status(self) -> bool:
        return self.status

stop_event = asyncio.Event()

dataq = queue.Queue()
pointq = queue.Queue()

# One of these controls the mouse
def mouser():
    oldpx = -1
    oldpy = -1

    while True:
        screen_xy, button, delay = pointq.get()
        print(f'Mouser: {pointq.qsize()}')

        # Same delay as the sample was read in,
        # shave off slightly so things don't get backed up
        # (and not too much that its noticeable)
        time.sleep((delay - 0.5) / 1000.0)

        # Manipulate mouse
        oldpx, oldpy = on_pose_solved(screen_xy, oldpx, oldpy)
        onButton(button)

        pointq.task_done()

# Each of these bad boyz runs the algorithm.
# Multiple of these bad boyz are needed since this is
# somewhat computationally intensive
def worker():
    while True:
        blobs, button, delay = dataq.get()
        print(f'Worker: {dataq.qsize()}')

        blobs = list(filter(lambda b: b[0] != 1023 and b[1] != 1023, blobs))
        print(blobs)

        rvec, tvec, screen_xy = solveController(blobs)

        if screen_xy is not None:
            pointq.put((screen_xy, button, delay))
        
        dataq.task_done()



def on_recv(sender: BleakGATTCharacteristic, data: bytearray):
    # print(f'From {sender}: {data}; Length: {len(data)}')
    # print(f'Parsed: {parse_packet(data)}')
    # print(f'Recieved packet! Queue length: {dataq.qsize()}')

    # Parse packet
    samples = parse_packet(data)
    if samples is None:
        log.warning("Malformed packet: %r", data)
        return
    

    for sample in samples:
        # print(sample)
        dataq.put(sample)

async def run_bluetooth():

    disconnect_event = asyncio.Event()
    connection = Connection()

    def on_disconnect(client: BLEDevice):
        print(f'Disconnected from {client.address}')
        connection.set_status(False)
        pyautogui.mouseUp()
        disconnect_event.set()

    async def on_connect(device: BLEDevice, advertising_data):
        if device.name == DEVICE_NAME and not connection.get_status():
            connection.set_status(True)
            async with BleakClient(device, disconnected_callback=on_disconnect) as client:
                print(f'Connected to {client.address}')
                await client.start_notify(DATA_CHARACTERISTIC_UUID, on_recv)
                await disconnect_event.wait()
        pass

    async with BleakScanner(on_connect) as scanner:
        await stop_event.wait()


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=mouser, daemon=True).start()
    try:
        asyncio.run(run_bluetooth())
    except KeyboardInterrupt:
        stop_event.set()
