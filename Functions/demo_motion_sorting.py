#!/usr/bin/env python3
# coding=utf8
"""
Demo: color-based sorting using PerceptionPipeline + MotionController.

Detects the largest block among red/green/blue, then uses MotionController
to pick and place it into its color bin.
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
from perception_pipeline import PerceptionPipeline
from motion_controller import MotionController


DRAW_RGB = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "None": (255, 255, 255),
}


def main():
    target_colors = ["red", "green", "blue"]
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
    chosen_color = "None"
    motion_thread = None  # background motion so camera stays live

    try:
        while True:
            frame = camera.frame
            if frame is None:
                continue

            out = frame.copy()
            pipeline.draw_crosshair(out)

            # Keep showing live camera while arm moves in background
            if motion_thread is not None and motion_thread.is_alive():
                cv2.putText(
                    out,
                    f"Sorting {chosen_color}...",
                    (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    DRAW_RGB.get(chosen_color, DRAW_RGB["None"]),
                    2,
                )
            elif motion_thread is not None and not motion_thread.is_alive():
                # Motion just finished
                motion_thread = None
            else:
                pre = pipeline.preprocess_frame(frame.copy())
                lab = pipeline.to_lab(pre)
                detection = pipeline.detect_largest_of_colors(lab, target_colors)

                if detection is not None:
                    color_name = detection["color"]
                    chosen_color = color_name
                    out = pipeline.annotate_detection(out, detection, DRAW_RGB[color_name])
                    world_x, world_y = detection["world"]
                    rect = detection["rect"]

                    distance = math.sqrt((world_x - last_x) ** 2 + (world_y - last_y) ** 2)
                    last_x, last_y = world_x, world_y

                    if distance < 0.3:
                        center_list.extend((world_x, world_y))
                        count += 1
                        if start_count_t1:
                            start_count_t1 = False
                            t1 = time.time()
                        if time.time() - t1 > 1.0 and count > 0:
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
                        f"Detected: {color_name}  ({world_x:.1f},{world_y:.1f})",
                        (10, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        DRAW_RGB[color_name],
                        2,
                    )
                else:
                    last_x, last_y = 0.0, 0.0
                    start_count_t1 = True
                    count = 0
                    center_list = []
                    chosen_color = "None"
                    cv2.putText(
                        out,
                        "No target detected",
                        (10, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        DRAW_RGB["None"],
                        2,
                    )

            cv2.imshow("Motion Sorting Demo", out)
            if cv2.waitKey(1) == 27:
                break

            # Start motion in background so camera continues updating.
            # Inside the background thread we first move above the block,
            # then re-localize its position and angle just before grasping.
            if picking and stable_point is not None and chosen_color != "None" and (
                motion_thread is None or not motion_thread.is_alive()
            ):
                approx_x, approx_y = stable_point
                color = chosen_color
                approx_rot = rotation_angle

                def do_motion():
                    # 1) Move above the approximate position
                    if not motion.move_above_block(approx_x, approx_y, y_offset=-2.0):
                        return

                    # 2) Short refinement loop: re-detect the same color and
                    #    update (x, y, angle) using the latest camera frames.
                    refined_x, refined_y, refined_rot = approx_x, approx_y, approx_rot
                    for _ in range(8):
                        frame_ref = camera.frame
                        if frame_ref is None:
                            time.sleep(0.02)
                            continue
                        pre_ref = pipeline.preprocess_frame(frame_ref.copy())
                        lab_ref = pipeline.to_lab(pre_ref)
                        det_ref = pipeline.detect_largest_of_colors(lab_ref, [color])
                        if det_ref is not None:
                            refined_x, refined_y = det_ref["world"]
                            refined_rot = det_ref["rect"][2]
                        time.sleep(0.02)

                    # 3) Grasp using refined pose, then place based on color
                    motion.grasp_block(refined_x, refined_y, refined_rot, z_override=motion.grasp_height)
                    drop_x, drop_y, drop_z = motion.get_drop_target(color)
                    motion.place_at(drop_x, drop_y, drop_z)

                motion_thread = threading.Thread(target=do_motion, daemon=True)
                motion_thread.start()
                picking = False
                stable_point = None

    finally:
        camera.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


