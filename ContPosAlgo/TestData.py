

"""
test_lightgun_processor.py
Dummy-data test suite for lightgun_processor.py
Run on any machine with OpenCV and NumPy installed - no hardware needed.

Usage:
    python test_lightgun_processor.py

Each test prints PASS or FAIL with details.
"""

import numpy as np
import sys
import math

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
    )
except ImportError as e:
    print(f"[ERROR] Could not import lightgun_processor: {e}")
    print("Make sure lightgun_processor.py is in the same directory.")
    sys.exit(1)

import cv2

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


# ===========================================================================
# Test 1 – parse_packet: valid full packet
# ===========================================================================
print("\n=== Test 1: parse_packet — valid 4-blob packet ===")
blobs, _ = parse_packet("100.5,200.3,700.0,150.8,680.2,600.1,120.9,590.4,1")
check("Returns 4 blobs", blobs is not None and len(blobs) == 4)
check("First blob x correct", abs(blobs[0][0] - 100.5) < 1e-6)
check("Fourth blob y correct", abs(blobs[3][1] - 590.4) < 1e-6)


# ===========================================================================
# Test 2 – parse_packet: partial visibility (2 blobs hidden)
# ===========================================================================
print("\n=== Test 2: parse_packet — 2 blobs hidden (-1,-1) ===")
blobs, _ = parse_packet("100.0,200.0,-1,-1,680.0,600.0,-1,-1,0")
check("Returns 2 visible blobs", blobs is not None and len(blobs) == 2)


# ===========================================================================
# Test 3 – parse_packet: malformed input
# ===========================================================================
print("\n=== Test 3: parse_packet — malformed / garbage input ===")
check("Empty string → None",     parse_packet("")[0] is None)
check("Too few values → None",   parse_packet("1,2,3")[0] is None)
check("Non-numeric → None",      parse_packet("a,b,c,d,e,f,g,h,0")[0] is None)
check("Trailing newline handled", parse_packet("10,20,30,40,50,60,70,80,0\n")[0] is not None)


# ===========================================================================
# Test 4 – sort_quad: correct spatial ordering
# ===========================================================================
print("\n=== Test 4: sort_quad — corner ordering ===")
# Deliberately shuffled order: bottom-right, top-left, bottom-left, top-right
shuffled = [(900, 700), (100, 100), (100, 700), (900, 100)]
sorted_pts = sort_quad(shuffled)
check("Top-left  is (100,100)", tuple(sorted_pts[0]) == (100.0, 100.0))
check("Top-right is (900,100)", tuple(sorted_pts[1]) == (900.0, 100.0))
check("Bot-right is (900,700)", tuple(sorted_pts[2]) == (900.0, 700.0))
check("Bot-left  is (100,700)", tuple(sorted_pts[3]) == (100.0, 700.0))


# ===========================================================================
# Test 5 – solveController: centred shot (gun pointing straight ahead)
# ===========================================================================
print("\n=== Test 5: solveController — centred, straight-on shot ===")
# Place LEDs 1.5 m in front of camera, centred
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
blobs = make_synthetic_blobs([tx, ty, tz])
rvec, tvec, screen_xy = solveController(blobs)

check("Solve succeeds", rvec is not None)
if rvec is not None:
    dist = float(np.linalg.norm(tvec))
    check("Distance roughly 1.5 m", abs(dist - 1.5) < 0.15,
          f"got {dist:.3f} m")
    x, y = screen_xy
    check("Aim X near centre (0.5)", abs(x - 0.5) < 0.05, f"x={x:.3f}")
    check("Aim Y near centre (0.5)", abs(y - 0.5) < 0.05, f"y={y:.3f}")


# ===========================================================================
# Test 6 – solveController: off-centre shot (top-left quadrant)
# ===========================================================================
print("\n=== Test 6: solveController — aiming top-left ===")
# Shift the LED array so centre projects into top-left of sensor
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
# Test 7 – solveController: only4 3 blobs visible
# ===========================================================================
print("\n=== Test 7: solveController — 3 blobs (1 occluded) ===")
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
blobs_4 = make_synthetic_blobs([tx, ty, tz])
blobs_3  = blobs_4[:3]   # drop fourth blob
rvec, tvec, screen_xy = solveController(blobs_3)
check("Solve still attempts with 3 blobs", True,  # just shouldn't crash
      "returned None is acceptable" if rvec is None else "got a result")


# ===========================================================================
# Test 8 – solveController: fewer than MIN_BLOBS → graceful failure
# ===========================================================================
print("\n=== Test 8: solveController — 1 blob (below minimum) ===")
rvec, tvec, screen_xy = solveController([(512.0, 384.0)])
check("Returns (None,None,None)", rvec is None and tvec is None and screen_xy is None)


# ===========================================================================
# Test 9 – solveController: slight rotational tilt
# ===========================================================================
print("\n=== Test 9: solveController — 5° yaw tilt ===")
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
    # Recovered yaw should be close to ground truth
    recovered_yaw_deg = math.degrees(float(rvec[1][0]))
    check("Yaw recovered within 2°", abs(recovered_yaw_deg - yaw_deg) < 2.0,
          f"gt={yaw_deg:.1f}° recovered={recovered_yaw_deg:.2f}°")


# ===========================================================================
# Test 10 – Full pipeline simulation (parse → solve)
# ===========================================================================
print("\n=== Test 10: End-to-end pipeline — serial line → aim point ===")
# Build a synthetic serial line for a centred shot
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
real_blobs = make_synthetic_blobs([tx, ty, tz])
# Serialise to the CSV format the ESP32 would send
csv_parts = []
for (bx, by) in real_blobs:
    csv_parts += [f"{bx:.2f}", f"{by:.2f}"]
serial_line = ",".join(csv_parts)

parsed = parse_packet(serial_line)
check("Packet parses correctly", parsed is not None and len(parsed) == 4)
rvec, tvec, screen_xy = solveController(parsed)
check("Pipeline produces aim point", screen_xy is not None)
if screen_xy:
    x, y = screen_xy
    check("Aim near screen centre", abs(x - 0.5) < 0.05 and abs(y - 0.5) < 0.05,
          f"x={x:.3f} y={y:.3f}")

# Quick mouse movement test
print("\n=== Mouse Movement Test ===")
print("Watch your mouse cursor - it should move to the centre of your screen")
import time
time.sleep(2)

tx = 1 / 2 #position adjustments made here
ty = 1 / 2 
tz = 1
mouse_blobs = make_synthetic_blobs([tx, ty, tz])
rvec, tvec, screen_xy = solveController(mouse_blobs)

from Algo import on_pose_solved
on_pose_solved(screen_xy)
print(f"Mouse moved to fraction: x={screen_xy[0]:.3f} y={screen_xy[1]:.3f}")


# ===========================================================================
# Test 11 – Button press parsing
# ===========================================================================
print("\n=== Test 11: parse_packet — button state ===")
blobs, button = parse_packet("100.5,200.3,700.0,150.8,680.2,600.1,120.9,590.4,0")
check("No button press parses correctly", button == 0)

blobs, button = parse_packet("100.5,200.3,700.0,150.8,680.2,600.1,120.9,590.4,1")
check("Button press parses correctly", button == 1)
check("Button press parses correctly", button == 1)

check("Blobs still parsed with button field", blobs is not None and len(blobs) == 4)

blobs, button = parse_packet("100.5,200.3,700.0,150.8,680.2,600.1,120.9,590.4")
check("Old 8-value packet rejected gracefully", blobs is None and button == 0)


# ===========================================================================
# Test 12 – Button press moves mouse and clicks
# ===========================================================================
print("\n=== Test 12: Button press — mouse click ===")
print("  Watch your mouse — it should click in 2 seconds...")
import time
time.sleep(2)

from Algo import on_pose_solved, onButton
import pyautogui

# Move to centre first
tx = -LED_SPACING_X / 2
ty = -LED_SPACING_Y / 2
tz = 1.5
blobs, _ = parse_packet("100.5,200.3,700.0,150.8,680.2,600.1,120.9,590.4,0")
blobs = make_synthetic_blobs([tx, ty, tz])
rvec, tvec, screen_xy = solveController(blobs)
on_pose_solved(screen_xy)

# Now fire the button
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

