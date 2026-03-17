#!/usr/bin/env python3
# coding=utf8
"""
Demo: basic pick-and-place using PerceptionPipeline + MotionController.

Detects a single target color and uses MotionController to pick one block
from the pickup area and place it into its color bin.
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import Camera
from perception_pipeline import PerceptionPipeline
from motion_controller import MotionController


DRAW_RGB = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "None": (255, 255, 255),
}


def main():
    target_color = "red"
    size = (640, 480)
    pipeline = PerceptionPipeline(size=size)
    motion = MotionController()
    camera = Camera.Camera()
    camera.camera_open()

    motion.init_pose()

    last_x, last_y = 0.0, 0.0
    center_list = []
    count = 0
    start_count_t1 = True
    t1 = 0.0
    picking = False
    stable_point = None
    rotation_angle = 0

    try:
        while True:
            frame = camera.frame
            if frame is None:
                continue

            out = frame.copy()
            pipeline.draw_crosshair(out)

            if picking:
                cv2.putText(
                    out,
                    "Picking up...",
                    (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    DRAW_RGB[target_color],
                    2,
                )
            else:
                pre = pipeline.preprocess_frame(frame.copy())
                lab = pipeline.to_lab(pre)
                detection = pipeline.detect_single_color(lab, target_color)

                if detection is not None:
                    out = pipeline.annotate_detection(out, detection, DRAW_RGB[target_color])
                    world_x, world_y = detection["world"]
                    rect = detection["rect"]

                    # Stability logic from ColorTracking: wait until block is still for 1.5s
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

            cv2.imshow("Motion Pick & Place Demo", out)
            if cv2.waitKey(1) == 27:
                break

            if picking and stable_point is not None:
                wx, wy = stable_point
                motion.pick_and_place_color(wx, wy, rotation_angle, target_color)
                picking = False
                stable_point = None

    finally:
        camera.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

