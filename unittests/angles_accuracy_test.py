import unittest
import numpy as np

from preprocessing.check_angles_accuracy_on_different_fps import calc_angle
from preprocessing.structs import Point

class TestAnglesAccuracy(unittest.TestCase):
    def test_calc_angle_1(self):
        p1 = Point(0, 0, 0)
        p2 = Point(1, 0, 0)
        p3 = Point(0, 1, 0)

        self.assertEqual(calc_angle(p2, p1, p3), 90)

    def test_calc_angle_2(self):
        p1 = Point(0, 0, 0)
        p2 = Point(5, 5, 5)
        p3 = Point(1, 2, 3)

        self.assertEqual(round(calc_angle(p2, p1, p3), 2), 22.21)

    def test_calc_angle_3(self):
        p1 = Point(0, 0, 0)
        p2 = Point(-7, -79, 3)
        p3 = Point(-120, 2, 3)

        self.assertEqual(round(calc_angle(p2, p1, p3), 2), 85.84)