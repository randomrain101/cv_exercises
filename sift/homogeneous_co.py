#%%
import sys
sys.path.append("./blatt_4")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

class Homogeneous3DPoint:
    def __init__(self, x, y, z, w=1):
        self.coordinates = np.array([x, y, z, w], dtype=float)

    def translation(self, tx, ty, tz):
        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1]], dtype=float)
        self.coordinates = translation_matrix @ self.coordinates

    def rotation_x(self, angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, cos_theta, -sin_theta, 0],
                                    [0, sin_theta, cos_theta, 0],
                                    [0, 0, 0, 1]], dtype=float)
        self.coordinates = rotation_matrix @ self.coordinates

    def rotation_y(self, angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([[cos_theta, 0, sin_theta, 0],
                                    [0, 1, 0, 0],
                                    [-sin_theta, 0, cos_theta, 0],
                                    [0, 0, 0, 1]], dtype=float)
        self.coordinates = rotation_matrix @ self.coordinates

    def rotation_z(self, angle):
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0, 0],
                                    [sin_theta, cos_theta, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=float)
        self.coordinates = rotation_matrix @ self.coordinates

    def affine(self, matrix):
        self.coordinates = matrix @ self.coordinates

    def scale(self, factor):
        scaling_matrix = np.array([[factor, 0, 0, 0],
                                   [0, factor, 0, 0],
                                   [0, 0, factor, 0],
                                   [0, 0, 0, 1]], dtype=float)
        self.coordinates = scaling_matrix @ self.coordinates

#%%
# Beispielanwendung
point = Homogeneous3DPoint(3, 7, 2)

# Verschiebung
point.translation(5, 2, 2)
print("Nach der Verschiebung:", point.coordinates)

# Rotation um die y-Achse
point.rotation_y(np.pi / 2)
print("Nach der Rotation um Y:", point.coordinates)

# Skalierung
point.scale(3)
print("Nach der Skalierung:", point.coordinates)

# %%
