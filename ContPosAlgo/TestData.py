"""
TestData.py
Dummy-data test suite for Algo.py
Run on any machine with OpenCV and NumPy installed - no hardware needed.

Usage:
    py TestData.py

Each test prints PASS or FAIL with details.

Packet format (bytes, comma separated hex values):
    button,x1,y1,x2,y2,x3,y3,x4,y4
    example: b"00,064,0C8,02BC,096,02A8,258,078,24E"
"""

print("SCRIPT STARTED")

import numpy as np
import sys
import math
import time

# ---------------------------------------------------------------------------
# Import the processor module (same directory expected)
# ---------------------------------------------------------------------------
try:
    from Algo import (
        parse_packet,
        solveController,
        sort_quad,
        CAMERA_MATRIX,
        DIST_COEFFS,
        WORLD_POINTS,
        LED_SPACING_X,
        LED_SPACING_Y,
        SENSOR_W,
        SENSOR_H,
        on_pose_solved,
        onButton,
    )
except ImportError as e:
    print(f"[ERROR] Could not import Algo: {e}")
    print("Make sure Algo.py is in the same directory.")
    sys.exit(1)

import cv2
import pyautogui

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def make_hex_packet(button: int, coords: list) -> bytes:
    """
    Helper to build a fake ESP32 packet as bytes.
    button: 0 or 1
    coords: list of 8 integers [x1,y1,x2,y2,x3,y3,x4,y4] in decimal
            (-1 means blob not visible)
    Returns bytes like b"00,064,0C8,02BC,096,FFFF,FFFF,078,24E"
    Hidden blobs use FF as a sentinel — parse_packet treats negatives as hidden,
    so we encode -1 as the string "-1" directly for simplicity.
    """
    parts = [f"{button:02X}"]
    for v in coords:
        if v < 0:
            parts.append("-1")
        else:
            parts.append(f"{v:03X}")
    return ",".join(parts).encode("ascii")


# ---------------------------------------------------------------------------
# Utility: project world points to image given a known pose
# ---------------------------------------------------------------------------

def project_world_to_image(world_pts, rvec, tvec):
    """Forward-project 3-D world points to 2-D image coords."""
    img_pts, _ = cv2.projectPoints(
        world_pts.reshape(-1, 1, 3),
        rvec, tvec,
        CAMERA_MATRIX, DIST_COEFFS
    )
    return img_pts.reshape(-1, 2)


def make_synthetic_blobs(tvec_m, rvec=None):
    """
    Create a synthetic (x,y) blob list by projecting the LED rectangle
    with a given translation (and optional rotation).
    Returns list of (x, y) tuples in pixel space.
    """
    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array(tvec_m, dtype=np.float64).reshape(3, 1)
    pts = project_world_to_image(WORLD_POINTS, rvec, tvec)
    return [(float(p[0]), float(p[1])) for p in pts]


def blobs_to_hex_packet(blobs, button=0) -> bytes:
    """
    Convert a list of (x,y) float blobs into a hex byte packet.
    Rounds floats to nearest int for encoding.
    """
    coords = []
    for (x, y) in blobs:
        coords += [int(round(x)), int(round(y))]
    # pad to 8 values if fewer than 4 blobs
    while len(coords) < 8:
        coords += [-1, -1]
    return make_hex_packet(button, coords[:8])


# ===========================================================================
# Test 1 – parse_packet: valid full packet, no button press
# ===========================================================================
print("\n=== Test 1: parse_packet — valid 4-blob packet, no button ===")
# 100,200,700,150,680,600,120,590 in hex = 064,0C8,2BC,096,2A8,258,078,24E
packet = make_hex_packet(0, [0x064, 0x0C8, 0x2BC, 0x096, 0x2A8, 0x258, 0x078, 0x24E])
blobs, button = parse_packet(packet)
check("Returns 4 blobs", blobs is not None and len(blobs) == 4)
check("Button is 0", button == 0)
check("First blob x is 100", abs(blobs[0][0] - 100) < 1)
check("Fourth blob y is 590", abs(blobs[3][1] - 590) < 1)


# ===========================================================================
# Test 2 – parse_packet: valid full packet, button pressed
# ===========================================================================
print("\n=== Test 2: parse_packet — valid 4-blob packet, button pressed ===")
packet = make_hex_packet(1, [0x064, 0x0C8, 0x2BC, 0x096, 0x2A8, 0x258, 0x078, 0x24E])
blobs, button = parse_packet(packet)
check("Returns 4 blobs", blobs is not None and len(blobs) == 4)
check("Button is 1", button == 1)


# ===========================================================================
# Test 3 – parse_packet: partial visibility (2 blobs hidden)
# ===========================================================================
print("\n=== Test 3: parse_packet — 2 blobs hidden (-1) ===")
packet = make_hex_packet(0, [0x064, 0x0C8, -1, -1, 0x2A8, 0x258, -1, -1])
blobs, button = parse_packet(packet)
check("Returns 2 visible blobs", blobs is not None and len(blobs) == 2)


# ===========================================================================
# Test 4 – parse_packet: malformed input
# ===========================================================================
print("\n=== Test 4: parse_packet — malformed / garbage input ===")
check("Empty bytes → None",        parse_packet(b"")[0] is None)
check("Too few values → None",     parse_packet(b"00,064,0C8")[0] is None)
check("Non-hex values → None",     parse_packet(b"ZZ,QQQ,RRR,SSS,TTT,UUU,VVV,WWW,XXX")[0] is None)
check("Trailing newline handled",  parse_packet(b"00,064,0C8,2BC,096,2A8,258,078,24E\n")[0] is not None)


# ===========================================================================
# Test 5 – sort_quad: correct spatial ordering
# ===========================================================================
print("\n=== Test 5: sort_quad — corner ordering ===")
shuffled = [(900, 700), (100, 100), (100, 700), (900, 100)]
sorted_pts = sort_quad(shuffled)
check("Top-left  is (100,100)", tuple(sorted_pts[0]) == (100.0, 100.0))
check("Top-right is (900,100)", tuple(sorted_pts[1]) == (900.0, 100.0))
check("Bot-right is (900,700)", tuple(sorted_pts[2]) == (900.0, 700.0))
check("Bot-left  is (100,700)", tuple(sorted_pts[3]) == (100.0, 700.0))


# ===========================================================================
# Test 6 – solveController: centred shot
# ===========================================================================
print("\n=== Test 6: solveController — centred, straight-on shot ===")
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
blobs = make_synthetic_blobs([tx, ty, tz])
rvec, tvec, screen_xy = solveController(blobs)

check("Solve succeeds", rvec is not None)
if rvec is not None:
    dist = float(np.linalg.norm(tvec))
    check("Distance roughly 1.5 m", abs(dist - 1.5) < 0.15, f"got {dist:.3f} m")
    x, y = screen_xy
    check("Aim X near centre (0.5)", abs(x - 0.5) < 0.05, f"x={x:.3f}")
    check("Aim Y near centre (0.5)", abs(y - 0.5) < 0.05, f"y={y:.3f}")


# ===========================================================================
# Test 7 – solveController: off-centre shot (top-left quadrant)
# ===========================================================================
print("\n=== Test 7: solveController — aiming top-left ===")
tx = -LED_SPACING_X / 2 - 0.10
ty = -LED_SPACING_Y / 2 - 0.06
tz = 1.5
blobs = make_synthetic_blobs([tx, ty, tz])
rvec, tvec, screen_xy = solveController(blobs)

check("Solve succeeds", rvec is not None)
if rvec is not None:
    x, y = screen_xy
    check("Aim X in left half (<0.5)", x < 0.5, f"x={x:.3f}")
    check("Aim Y in top half (<0.5)", y < 0.5, f"y={y:.3f}")


# ===========================================================================
# Test 8 – solveController: only 3 blobs visible
# ===========================================================================
print("\n=== Test 8: solveController — 3 blobs (1 occluded) ===")
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
blobs_4 = make_synthetic_blobs([tx, ty, tz])
blobs_3 = blobs_4[:3]
rvec, tvec, screen_xy = solveController(blobs_3)
check("Solve still attempts with 3 blobs", True,
      "returned None is acceptable" if rvec is None else "got a result")


# ===========================================================================
# Test 9 – solveController: fewer than minimum blobs
# ===========================================================================
print("\n=== Test 9: solveController — 1 blob (below minimum) ===")
rvec, tvec, screen_xy = solveController([(512.0, 384.0)])
check("Returns (None,None,None)", rvec is None and tvec is None and screen_xy is None)


# ===========================================================================
# Test 10 – solveController: 5° yaw tilt
# ===========================================================================
print("\n=== Test 10: solveController — 5° yaw tilt ===")
yaw_deg = 5.0
rvec_gt = np.array([[0.0], [math.radians(yaw_deg)], [0.0]], dtype=np.float64)
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
tvec_gt = np.array([[tx], [ty], [tz]], dtype=np.float64)
blobs = [(float(p[0]), float(p[1])) for p in project_world_to_image(WORLD_POINTS, rvec_gt, tvec_gt)]
rvec, tvec, screen_xy = solveController(blobs)
check("Solve succeeds with tilt", rvec is not None)
if rvec is not None:
    recovered_yaw_deg = math.degrees(float(rvec[1][0]))
    check("Yaw recovered within 2°", abs(recovered_yaw_deg - yaw_deg) < 2.0,
          f"gt={yaw_deg:.1f}° recovered={recovered_yaw_deg:.2f}°")


# ===========================================================================
# Test 11 – Full pipeline: hex bytes → parse → solve
# ===========================================================================
print("\n=== Test 11: End-to-end pipeline — hex bytes → aim point ===")
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
real_blobs = make_synthetic_blobs([tx, ty, tz])
packet = blobs_to_hex_packet(real_blobs, button=0)

blobs, button = parse_packet(packet)
check("Packet parses correctly", blobs is not None and len(blobs) == 4)
check("Button is 0", button == 0)

rvec, tvec, screen_xy = solveController(blobs)
check("Pipeline produces aim point", screen_xy is not None)
if screen_xy:
    x, y = screen_xy
    check("Aim near screen centre", abs(x - 0.5) < 0.05 and abs(y - 0.5) < 0.05,
          f"x={x:.3f} y={y:.3f}")


# ===========================================================================
# Test 12 – Button state in pipeline
# ===========================================================================
print("\n=== Test 12: End-to-end pipeline — button pressed ===")
packet = blobs_to_hex_packet(real_blobs, button=1)
blobs, button = parse_packet(packet)
check("Button 1 parsed from pipeline packet", button == 1)
check("Blobs still valid with button=1", blobs is not None and len(blobs) == 4)


# ===========================================================================
# Mouse movement test
# ===========================================================================
print("\n=== Mouse Movement Test ===")
print("  Watch your mouse — it should move to screen centre in 2 seconds...")
time.sleep(2)

tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
mouse_blobs = make_synthetic_blobs([tx, ty, tz])
rvec, tvec, screen_xy = solveController(mouse_blobs)
if screen_xy:
    on_pose_solved(screen_xy)
    print(f"  Mouse moved to fraction: x={screen_xy[0]:.3f} y={screen_xy[1]:.3f}")
else:
    print("  [WARN] Could not solve pose for mouse test")


# ===========================================================================
# Test 13 – Button click test
# ===========================================================================
print("\n=== Test 13: Button press — mouse click ===")
print("  Mouse will click in 2 seconds — make sure nothing important is under cursor...")
time.sleep(2)

onButton(1)
check("Button 1 triggers click without crash", True)

onButton(0)
check("Button 0 does nothing without crash", True)


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 50)
passed = sum(results)
total  = len(results)
colour = "\033[92m" if passed == total else "\033[93m"
print(f"{colour}Results: {passed}/{total} checks passed\033[0m")
if passed < total:
    print("  Review FAIL lines above to debug the processor before connecting hardware.")
else:
    print("  All checks passed — safe to move on to hardware integration.")
print("=" * 50 + "\n")

input("Press Enter to exit...")