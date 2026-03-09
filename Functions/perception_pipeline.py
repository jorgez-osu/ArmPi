#!/usr/bin/env python3
# coding=utf8
"""
Reusable perception pipeline extracted from ColorTracking/Sorting/Palletizing.
"""

import time
import math
from collections import deque

import cv2
import numpy as np

from LABConfig import color_range
from ArmIK.Transform import convertCoordinate, getCenter, getMaskROI, getROI
from CameraCalibration.CalibrationConfig import square_length


class PerceptionPipeline:
    def __init__(self, size=(640, 480), min_valid_contour_area=300, min_detect_area=2500):
        self.size = size
        self.min_valid_contour_area = min_valid_contour_area
        self.min_detect_area = min_detect_area
        self.reset_tracking_state()

    def reset_tracking_state(self):
        self.last_x = 0.0
        self.last_y = 0.0
        self.center_samples = []
        self.timer_started = False
        self.t_start = 0.0
        self.vote_window = deque(maxlen=3)

    def draw_crosshair(self, image):
        h, w = image.shape[:2]
        cv2.line(image, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 200), 1)
        cv2.line(image, (int(w / 2), 0), (int(w / 2), h), (0, 0, 200), 1)

    def preprocess_frame(self, image):
        frame_resize = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(frame_resize, (11, 11), 11)

    def apply_roi_if_present(self, frame, roi):
        if roi is None:
            return frame
        return getMaskROI(frame, roi, self.size)

    def to_lab(self, frame_bgr):
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    def largest_contour(self, contours):
        area_max_contour = None
        contour_area_max = 0
        for contour in contours:
            contour_area = math.fabs(cv2.contourArea(contour))
            if contour_area > contour_area_max:
                contour_area_max = contour_area
                if contour_area > self.min_valid_contour_area:
                    area_max_contour = contour
        return area_max_contour, contour_area_max

    def _segment_color(self, frame_lab, color_name):
        frame_mask = cv2.inRange(frame_lab, color_range[color_name][0], color_range[color_name][1])
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        return self.largest_contour(contours)

    def _build_detection(self, contour, area, color_name):
        if contour is None or area <= self.min_detect_area:
            return None

        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        roi = getROI(box)
        img_centerx, img_centery = getCenter(rect, roi, self.size, square_length)
        world_x, world_y = convertCoordinate(img_centerx, img_centery, self.size)

        return {
            "color": color_name,
            "area": area,
            "rect": rect,
            "box": box,
            "roi": roi,
            "img_center": (img_centerx, img_centery),
            "world": (world_x, world_y),
        }

    def detect_single_color(self, frame_lab, color_name):
        contour, area = self._segment_color(frame_lab, color_name)
        return self._build_detection(contour, area, color_name)

    def detect_largest_of_colors(self, frame_lab, target_colors):
        best = None
        for color_name in target_colors:
            contour, area = self._segment_color(frame_lab, color_name)
            detection = self._build_detection(contour, area, color_name)
            if detection is None:
                continue
            if best is None or detection["area"] > best["area"]:
                best = detection
        return best

    def update_stability(self, world_x, world_y, distance_threshold=0.5, stable_seconds=1.0):
        distance = math.sqrt((world_x - self.last_x) ** 2 + (world_y - self.last_y) ** 2)
        self.last_x, self.last_y = world_x, world_y

        stable = False
        avg_world = None
        if distance < distance_threshold:
            self.center_samples.append((world_x, world_y))
            if not self.timer_started:
                self.timer_started = True
                self.t_start = time.time()
            if time.time() - self.t_start > stable_seconds and self.center_samples:
                stable = True
                avg_world = tuple(np.mean(np.array(self.center_samples), axis=0))
                self.center_samples = []
                self.timer_started = True
                self.t_start = time.time()
        else:
            self.center_samples = []
            self.timer_started = True
            self.t_start = time.time()
        return distance, stable, avg_world

    def update_color_vote(self, color_name):
        code_map = {"red": 1, "green": 2, "blue": 3}
        self.vote_window.append(code_map.get(color_name, 0))
        if len(self.vote_window) < self.vote_window.maxlen:
            return None
        avg_code = int(round(np.mean(np.array(self.vote_window))))
        self.vote_window.clear()
        inv_map = {1: "red", 2: "green", 3: "blue"}
        return inv_map.get(avg_code, "None")

    def annotate_detection(self, image, detection, draw_color):
        if detection is None:
            return image
        cv2.drawContours(image, [detection["box"]], -1, draw_color, 2)
        wx, wy = detection["world"]
        anchor = (min(detection["box"][0, 0], detection["box"][2, 0]), detection["box"][2, 1] - 10)
        cv2.putText(
            image,
            f"({wx},{wy})",
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            draw_color,
            1,
        )
        return image
