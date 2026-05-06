"""
Kalman filter implementation for 2D translation tracking.This module tracks object position and velocity in image coordinates using a
constant-velocity state model. It is used in the parachute tracking pipeline
to predict centroid motion across frames.
"""

import numpy as np


class TranslationKalmanFilter:
    
    # Here the Kalman filter is for 2D translation with state [x, y, vx, vy].


    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=5.0):
        
        # ToInitialize translation Kalman filter matrices and parameters.
        self.dt = float(dt)

        # State transition matrix for constant velocity:
        # x_t = x_(t-1) + vx*dt, y_t = y_(t-1) + vy*dt.
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Measurement matrix: only position is observed from segmentation.
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        # Process noise covariance controls motion model flexibility.
        self.Q = float(process_noise) * np.array(
            [
                [self.dt**4 / 4.0, 0.0, self.dt**3 / 2.0, 0.0],
                [0.0, self.dt**4 / 4.0, 0.0, self.dt**3 / 2.0],
                [self.dt**3 / 2.0, 0.0, self.dt**2, 0.0],
                [0.0, self.dt**3 / 2.0, 0.0, self.dt**2],
            ],
            dtype=float,
        )

        # Measurement noise covariance models centroid detection noise.
        self.R = float(measurement_noise) * np.eye(2, dtype=float)

        # Initial state covariance: high uncertainty before enough observations.
        self.P = np.eye(4, dtype=float) * 1000.0

        # Identity matrix used in covariance update equation.
        self.I = np.eye(4, dtype=float)

        # Initial state vector [x, y, vx, vy]^T.
        self.x = np.zeros((4, 1), dtype=float)
        self.initialized = False

    def initialize(self, initial_x, initial_y):
    
        #Initialize filter state from first centroid measurement.

        self.x = np.array(
            [[float(initial_x)], [float(initial_y)], [0.0], [0.0]],
            dtype=float,
        )
        self.initialized = True

    def predict(self):
        
        #Perform the Kalman prediction step.

        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling predict().")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement_x, measurement_y):
        
        # Here it perform the Kalman measurement update step.

        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling update().")

        z = np.array([[float(measurement_x)], [float(measurement_y)]], dtype=float)

        # Innovation/residual between measurement and prediction.
        y = z - (self.H @ self.x)

        # Innovation covariance combines predicted uncertainty and measurement noise.
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain controls how strongly we trust the new measurement.
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Correct state and covariance with current measurement.
        self.x = self.x + (K @ y)
        self.P = (self.I - (K @ self.H)) @ self.P
        return self.x

    def get_position(self):
        
        #Return the current estimated object position.

        return float(self.x[0, 0]), float(self.x[1, 0])
