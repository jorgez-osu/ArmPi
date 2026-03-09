#!/usr/bin/env python3
# coding=utf8
"""
Demo for extra functionality from ColorSorting/ColorPalletizing:
1) choose largest contour among target colors
2) vote color across frames for robustness
3) stability check before "ready to pick"
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
    target_colors = ("red", "green", "blue")
    pipeline = PerceptionPipeline(size=(640, 480))
    camera = Camera.Camera()
    camera.camera_open()

    roi = None
    show_color = "None"
    ready_to_pick = False
    stable_point = None

    try:
        while True:
            frame = camera.frame
            if frame is None:
                continue

            out = frame.copy()
            pipeline.draw_crosshair(out)

            pre = pipeline.preprocess_frame(frame.copy())
            pre = pipeline.apply_roi_if_present(pre, roi)
            lab = pipeline.to_lab(pre)
            detection = pipeline.detect_largest_of_colors(lab, target_colors)

            if detection is not None:
                roi = detection["roi"]
                voted = pipeline.update_color_vote(detection["color"])
                if voted is not None:
                    show_color = voted
                draw_color = DRAW_RGB.get(show_color, DRAW_RGB["None"])
                out = pipeline.annotate_detection(out, detection, draw_color)

                wx, wy = detection["world"]
                distance, stable, avg_world = pipeline.update_stability(
                    wx, wy, distance_threshold=0.5, stable_seconds=1.0
                )
                if stable:
                    ready_to_pick = True
                    stable_point = avg_world
                else:
                    ready_to_pick = False

                cv2.putText(
                    out,
                    f"Detected: {detection['color']}  Voted: {show_color}",
                    (10, out.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    draw_color,
                    2,
                )
                cv2.putText(
                    out,
                    f"Motion delta: {distance:.3f}",
                    (10, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
            else:
                roi = None
                show_color = "None"
                ready_to_pick = False
                cv2.putText(
                    out,
                    "No color block detected",
                    (10, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    DRAW_RGB["None"],
                    2,
                )

            if ready_to_pick and stable_point is not None:
                cv2.putText(
                    out,
                    f"READY TO PICK at ({round(stable_point[0],2)}, {round(stable_point[1],2)})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Perception Multi-color Demo", out)
            if cv2.waitKey(1) == 27:
                break
    finally:
        camera.camera_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
