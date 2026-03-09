#!/usr/bin/env python3
# coding=utf8
"""
Simple demo: locate a block in pickup area and label coordinates.
"""

import cv2

import Camera
from perception_pipeline import PerceptionPipeline


DRAW_RGB = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "None": (255, 255, 255),
}


def main():
    target_color = "red"
    pipeline = PerceptionPipeline(size=(640, 480))
    camera = Camera.Camera()
    camera.camera_open()

    try:
        while True:
            frame = camera.frame
            if frame is None:
                continue

            out = frame.copy()
            pipeline.draw_crosshair(out)
            pre = pipeline.preprocess_frame(frame.copy())
            lab = pipeline.to_lab(pre)
            detection = pipeline.detect_single_color(lab, target_color)

            if detection is not None:
                out = pipeline.annotate_detection(out, detection, DRAW_RGB[target_color])
                world_x, world_y = detection["world"]
                cv2.putText(
                    out,
                    f"Target: {target_color}  World: ({world_x}, {world_y})",
                    (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    DRAW_RGB[target_color],
                    2,
                )
            else:
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
    finally:
        camera.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
