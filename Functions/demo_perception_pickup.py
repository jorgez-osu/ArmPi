#!/usr/bin/env python3
# coding=utf8
"""
Demo: locate a block in the pickup area, wait for stability, then pick and place it.

Targets a single color only (default: red). If multiple cubes are in view,
only the one matching the target color is detected and picked; others are ignored.
Picks one block at a time.
"""

import sys
import os
import time
import math
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import Camera
import HiwonderSDK.Board as Board
from ArmIK.ArmMoveIK import ArmIK
from ArmIK.Transform import getAngle
from perception_pipeline import PerceptionPipeline


# Gripper closed position when grasping
SERVO1 = 500

# Match ColorTracking.py: approach above block, then lower to grasp height.
APPROACH_HEIGHT_CM = 5
GRASP_HEIGHT_CM = 2
# Offsets to correct for calibration/setup. If gripper aims too far in +Y, set Y_OFFSET_CM < 0 (e.g. -2.5 for ~1 inch).
# If gripper needs to go lower in Z, set Z_OFFSET_CM < 0 (e.g. -0.5).
Y_OFFSET_CM = 0
Z_OFFSET_CM = -0.5   # slightly lower grasp in Z (tune if needed)

# Drop coordinates (x, y, z) per color
DROP_COORDS = {
    "red": (-15 + 0.5, 12 - 0.5, 1.5),
    "green": (-15 + 0.5, 6 - 0.5, 1.5),
    "blue": (-15 + 0.5, 0 - 0.5, 1.5),
}

DRAW_RGB = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "None": (255, 255, 255),
}


def init_arm(ak):
    """Move arm to initial pose."""
    Board.setBusServoPulse(1, SERVO1 - 50, 300)
    Board.setBusServoPulse(2, 500, 500)
    ak.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)


def set_buzzer(duration=0.1):
    Board.setBuzzer(0)
    Board.setBuzzer(1)
    time.sleep(duration)
    Board.setBuzzer(0)


def run_pickup_sequence(ak, world_x, world_y, rotation_angle, target_color):
    """Execute pick-and-place for block at (world_x, world_y)."""
    coord = DROP_COORDS.get(target_color, DROP_COORDS["red"])

    # Approach above block (same as ColorTracking: Y-2 for approach)
    wy_approach = world_y - 2 + Y_OFFSET_CM
    wz_grasp = GRASP_HEIGHT_CM + Z_OFFSET_CM
    result = ak.setPitchRangeMoving((world_x, wy_approach, APPROACH_HEIGHT_CM), -90, -90, 0)
    if result is False:
        return False
    time.sleep(result[2] / 1000)

    # Open gripper and rotate
    Board.setBusServoPulse(1, SERVO1 - 280, 500)
    servo2_angle = getAngle(world_x, world_y + Y_OFFSET_CM, rotation_angle)
    Board.setBusServoPulse(2, servo2_angle, 500)
    time.sleep(0.8)

    # Lower and grasp (same as ColorTracking: (world_X, world_Y, 2))
    ak.setPitchRangeMoving((world_x, world_y + Y_OFFSET_CM, wz_grasp), -90, -90, 0, 1000)
    time.sleep(2)
    Board.setBusServoPulse(1, SERVO1, 500)
    time.sleep(1)

    # Lift
    Board.setBusServoPulse(2, 500, 500)
    ak.setPitchRangeMoving((world_x, world_y + Y_OFFSET_CM, 12), -90, -90, 0, 1000)
    time.sleep(1)

    # Move to drop position
    result = ak.setPitchRangeMoving((coord[0], coord[1], 12), -90, -90, 0)
    time.sleep(result[2] / 1000)
    servo2_angle = getAngle(coord[0], coord[1], -90)
    Board.setBusServoPulse(2, servo2_angle, 500)
    time.sleep(0.5)

    # Lower and release
    ak.setPitchRangeMoving((coord[0], coord[1], coord[2] + 3), -90, -90, 0, 500)
    time.sleep(0.5)
    ak.setPitchRangeMoving(coord, -90, -90, 0, 1000)
    time.sleep(0.8)
    Board.setBusServoPulse(1, SERVO1 - 200, 500)
    time.sleep(0.8)

    # Return to home
    ak.setPitchRangeMoving((coord[0], coord[1], 12), -90, -90, 0, 800)
    time.sleep(0.8)
    init_arm(ak)
    time.sleep(1.5)
    return True


def main():
    target_color = "red"
    size = (640, 480)  # Match ColorTracking / ColorSorting
    pipeline = PerceptionPipeline(size=size)
    ak = ArmIK()
    camera = Camera.Camera()
    camera.camera_open()

    init_arm(ak)
    set_buzzer(0.1)

    picking = False
    stable_point = None
    rotation_angle = 0
    pickup_thread = None  # when set, pickup runs in background so camera keeps updating
    # Same stability/averaging as ColorTracking: accumulate (world_x, world_y) when distance < 0.3 for 1.5s, then use mean
    center_list = []
    count = 0
    last_x, last_y = 0.0, 0.0
    start_count_t1 = True
    t1 = 0.0

    try:
        while True:
            try:
                frame = camera.frame
                if frame is None:
                    continue

                out = frame.copy()
                pipeline.draw_crosshair(out)

                # Show live camera during pickup so arm motion is visible when recording
                if pickup_thread is not None and pickup_thread.is_alive():
                    cv2.putText(
                        out,
                        "Picking up...",
                        (10, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        DRAW_RGB[target_color],
                        2,
                    )
                elif pickup_thread is not None and not pickup_thread.is_alive():
                    # Pickup just finished
                    pickup_thread = None
                    set_buzzer(0.1)
                else:
                    pre = pipeline.preprocess_frame(frame.copy())
                    lab = pipeline.to_lab(pre)
                    detection = pipeline.detect_single_color(lab, target_color)

                    if detection is not None:
                        out = pipeline.annotate_detection(out, detection, DRAW_RGB[target_color])
                        world_x, world_y = detection["world"]
                        rect = detection["rect"]

                        # ColorTracking logic: distance from last position, accumulate when stable
                        distance = math.sqrt((world_x - last_x) ** 2 + (world_y - last_y) ** 2)
                        last_x, last_y = world_x, world_y

                        if distance < 0.3:
                            center_list.extend((world_x, world_y))
                            count += 1
                            if start_count_t1:
                                start_count_t1 = False
                                t1 = time.time()
                            if time.time() - t1 > 1.5 and count > 0:
                                rotation_angle = rect[2]
                                start_count_t1 = True
                                world_X, world_Y = np.mean(np.array(center_list).reshape(count, 2), axis=0)
                                stable_point = (world_X, world_Y)
                                center_list = []
                                count = 0
                                picking = True
                        else:
                            t1 = time.time()
                            start_count_t1 = True
                            count = 0
                            center_list = []

                        cv2.putText(
                            out,
                            f"Target: {target_color}  World: ({world_x:.1f}, {world_y:.1f})",
                            (10, out.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            DRAW_RGB[target_color],
                            2,
                        )
                    else:
                        last_x, last_y = 0.0, 0.0
                        start_count_t1 = True
                        count = 0
                        center_list = []
                        cv2.putText(
                            out,
                            f"Target: {target_color}  Not detected",
                            (10, out.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            DRAW_RGB["None"],
                            2,
                        )

                cv2.imshow("Perception Pickup Demo", out)
                if cv2.waitKey(1) == 27:
                    break

                # Start pickup in background thread so camera keeps updating (arm visible on screen)
                if picking and stable_point is not None and (pickup_thread is None or not pickup_thread.is_alive()):
                    wx, wy = stable_point
                    rot = rotation_angle
                    def do_pickup():
                        run_pickup_sequence(ak, wx, wy, rot, target_color)
                    pickup_thread = threading.Thread(target=do_pickup, daemon=True)
                    pickup_thread.start()
                    picking = False
                    stable_point = None
            except KeyboardInterrupt:
                break

    finally:
        try:
            camera.camera_close()
            time.sleep(0.5)  # Let camera thread see closed state before destroying windows
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
