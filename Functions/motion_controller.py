#!/usr/bin/env python3
# coding=utf8
"""
MotionController: high-level motion primitives for ArmPi.

This wraps the manufacturer motion logic from ColorTracking, ColorSorting,
and ColorPalletizing into a readable Python class that can be combined
with perception code (e.g. PerceptionPipeline) for pick-and-place,
sorting, and stacking.
"""

import time

import HiwonderSDK.Board as Board
from ArmIK.ArmMoveIK import ArmIK
from ArmIK.Transform import getAngle


class MotionController:
    def __init__(self, drop_coords=None, grasp_height=2.0, approach_height=5.0,
                 x_offset_cm=1.0, y_offset_cm=1.5, z_offset_cm=-2.0):
        """
        drop_coords: dict like {"red": (x, y, z), "green": (...), "blue": (...)}
        grasp_height: Z (cm) when closing gripper on block.
        approach_height: Z (cm) used when approaching above block.
        x_offset_cm: add to world_x for pick (positive = right).
        y_offset_cm: add to world_y for pick (positive = higher in Y).
        z_offset_cm: add to grasp z (negative = gripper reaches lower before grasp).
        """
        self.ak = ArmIK()
        self.servo_grip_closed = 500
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.x_offset_cm = x_offset_cm
        self.y_offset_cm = y_offset_cm
        self.z_offset_cm = z_offset_cm

        # Default drop coordinates (from ColorTracking / ColorSorting)
        self.drop_coords = drop_coords or {
            "red": (-15 + 0.5, 12 - 0.5, 1.5),
            "green": (-15 + 0.5, 6 - 0.5, 1.5),
            "blue": (-15 + 0.5, 0 - 0.5, 1.5),
        }

        # Stacking base coordinates (from ColorPalletizing)
        self.stack_base = {
            "red": (-15 + 1, -7 - 0.5, 1.5),
            "green": (-15 + 1, -7 - 0.5, 1.5),
            "blue": (-15 + 1, -7 - 0.5, 1.5),
        }
        self.stack_dz = 2.5  # height step between blocks
        self.stack_level = {"red": 0, "green": 0, "blue": 0}

    # --- Basic helpers -------------------------------------------------

    def init_pose(self):
        """Move arm to initial pose (same as initMove in manufacturer code)."""
        Board.setBusServoPulse(1, self.servo_grip_closed - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.ak.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

    def open_gripper(self):
        Board.setBusServoPulse(1, self.servo_grip_closed - 280, 500)

    def close_gripper(self):
        Board.setBusServoPulse(1, self.servo_grip_closed, 500)

    def rotate_gripper_towards(self, x, y, block_angle):
        """Rotate wrist to align with block at world (x, y) and angle."""
        servo2_angle = getAngle(x, y, block_angle)
        Board.setBusServoPulse(2, servo2_angle, 500)
        return servo2_angle

    def lift_above(self, x, y, z=12):
        """Lift arm above (x, y) to height z."""
        self.ak.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)

    # --- Primitive motions ---------------------------------------------

    def move_above_block(self, world_x, world_y, y_offset=-2.0):
        """
        Move above the detected block, using Y-2 and approach height.
        Applies x_offset_cm and y_offset_cm to world position.
        """
        target_x = world_x + self.x_offset_cm
        target_y = world_y + self.y_offset_cm + y_offset
        result = self.ak.setPitchRangeMoving(
            (target_x, target_y, self.approach_height), -90, -90, 0
        )
        if result is False:
            return False
        time.sleep(result[2] / 1000.0)
        return True

    def grasp_block(self, world_x, world_y, block_angle, z_override=None, y_offset=0.0):
        """
        Full grasp sequence at (world_x, world_y):
        - open gripper
        - rotate wrist
        - lower to grasp height
        - close gripper
        - lift to 12 cm
        Applies x_offset_cm, y_offset_cm, z_offset_cm. Returns True on success.
        """
        base_z = self.grasp_height if z_override is None else z_override
        z = base_z + self.z_offset_cm  # e.g. -2 = 2 cm lower
        x = world_x + self.x_offset_cm
        y = world_y + self.y_offset_cm + y_offset

        # Open and rotate
        self.open_gripper()
        self.rotate_gripper_towards(x, y, block_angle)
        time.sleep(0.8)

        # Lower and close
        self.ak.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(1.5)
        self.close_gripper()
        time.sleep(0.8)

        # Lift
        Board.setBusServoPulse(2, 500, 500)
        self.lift_above(x, y, 12)
        time.sleep(1.0)
        return True

    def place_at(self, x, y, z):
        """
        Place currently held block at (x, y, z) and return to initial pose.
        """
        # Move above drop position
        result = self.ak.setPitchRangeMoving((x, y, 12), -90, -90, 0)
        if result is False:
            return False
        time.sleep(result[2] / 1000.0)

        # Rotate gripper to face stack/drop location
        self.rotate_gripper_towards(x, y, -90)
        time.sleep(0.5)

        # Lower, open, and lift
        self.ak.setPitchRangeMoving((x, y, z + 3), -90, -90, 0, 500)
        time.sleep(0.5)
        self.ak.setPitchRangeMoving((x, y, z), -90, -90, 0, 1000)
        time.sleep(0.8)
        Board.setBusServoPulse(1, self.servo_grip_closed - 200, 500)
        time.sleep(0.8)
        self.lift_above(x, y, 12)
        time.sleep(0.8)
        self.init_pose()
        time.sleep(1.5)
        return True

    # --- High-level tasks & helpers ------------------------------------

    def get_drop_target(self, color_name):
        """Return (x, y, z) drop target for sorting."""
        if color_name not in self.drop_coords:
            color_name = "red"
        return self.drop_coords[color_name]

    def get_next_stack_target(self, color_name):
        """
        Return (x, y, z) for the next stack level for this color and
        update internal stack_level, matching ColorPalletizing behavior.
        """
        if color_name not in self.stack_base:
            color_name = "red"

        level = self.stack_level[color_name]
        base_x, base_y, base_z = self.stack_base[color_name]
        target_z = base_z + level * self.stack_dz
        # Cycle stack height after 3 blocks
        level = (level + 1) % 3
        self.stack_level[color_name] = level
        return base_x, base_y, target_z

    def pick_and_place_color(self, world_x, world_y, block_angle, color_name):
        """
        Basic pick-and-place:
        - move above detected block
        - grasp
        - place into color-specific drop bin
        """
        drop_x, drop_y, drop_z = self.get_drop_target(color_name)

        if not self.move_above_block(world_x, world_y, y_offset=-2.0):
            return False

        self.grasp_block(world_x, world_y, block_angle, z_override=self.grasp_height)
        self.place_at(drop_x, drop_y, drop_z)
        return True

    def sort_block(self, world_x, world_y, block_angle, color_name):
        """
        Color sorting: identical to pick_and_place_color today, but
        kept separate to allow different strategies later.
        """
        return self.pick_and_place_color(world_x, world_y, block_angle, color_name)

    def stack_block(self, world_x, world_y, block_angle, color_name):
        """
        Block stacking (ColorPalletizing behavior):
        - pick block from world_x, world_y
        - compute next stack level for color
        - place at base + level*dz
        """
        if color_name not in self.stack_base:
            color_name = "red"

        # Compute stack height for this color
        level = self.stack_level[color_name]
        base_x, base_y, base_z = self.stack_base[color_name]
        target_z = base_z + level * self.stack_dz
        # Cycle stack height after 3 blocks, similar to manufacturer code
        level = (level + 1) % 3
        self.stack_level[color_name] = level

        if not self.move_above_block(world_x, world_y, y_offset=-2.0):
            return False

        self.grasp_block(world_x, world_y, block_angle, z_override=2.0)
        self.place_at(base_x, base_y, target_z)
        return True

