"""
Kalman filter implementation for rotation/orientation tracking.

This module tracks object orientation angle and angular velocity using a
constant-angular-velocity state model. It includes angle normalization to
handle wrap-around behavior in orientation measurements.
"""

import numpy as np


class RotationKalmanFilter:
    """
    Kalman filter for orientation tracking with state [theta, omega].

    State vector:
        [theta, omega]^T
        theta: orientation angle in degrees
        omega: angular velocity in degrees per frame

    Measurement vector:
        [theta]^T
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=5.0):
        """
        Initialize rotation Kalman filter matrices and parameters.

        Args:
            dt (float): Time step between frames.
            process_noise (float): Process uncertainty scale.
            measurement_noise (float): Measurement uncertainty scale.
        """
        self.dt = float(dt)

        # State transition matrix for constant angular velocity:
        # theta_t = theta_(t-1) + omega*dt, omega_t = omega_(t-1).
        self.F = np.array(
            [
                [1.0, self.dt],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        # Measurement matrix: only orientation angle theta is observed.
        self.H = np.array([[1.0, 0.0]], dtype=float)

        # Process noise covariance controls model flexibility over time.
        self.Q = float(process_noise) * np.array(
            [
                [self.dt**4 / 4.0, self.dt**3 / 2.0],
                [self.dt**3 / 2.0, self.dt**2],
            ],
            dtype=float,
        )

        # Measurement noise covariance models orientation extraction noise.
        self.R = np.array([[float(measurement_noise)]], dtype=float)

        # Initial covariance uses high uncertainty before filtering stabilizes.
        self.P = np.eye(2, dtype=float) * 1000.0

        # Identity matrix used in covariance update.
        self.I = np.eye(2, dtype=float)

        # State vector [theta, omega]^T initialized at zero.
        self.x = np.zeros((2, 1), dtype=float)
        self.initialized = False

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-90, 90] degrees.

        Why normalization is required:
            Orientation values wrap around boundaries, so this keeps angles in
            a consistent range and avoids false large differences.

        Args:
            angle (float): Input angle in degrees.

        Returns:
            float: Normalized angle in [-90, 90].
        """
        normalized = ((float(angle) + 90.0) % 180.0) - 90.0
        return normalized

    def initialize(self, initial_theta):
        """
        Initialize filter state from first orientation measurement.

        Angular velocity starts at zero and is learned during updates.

        Args:
            initial_theta (float): Initial measured orientation in degrees.
        """
        theta0 = self.normalize_angle(initial_theta)
        self.x = np.array([[theta0], [0.0]], dtype=float)
        self.initialized = True

    def predict(self):
        """
        Perform the Kalman prediction step for orientation state.

        Prediction equations:
            x = F x
            P = F P F^T + Q

        Returns:
            np.ndarray: Predicted state vector with shape (2, 1).
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling predict().")

        # Predict next angle/angular velocity from motion model.
        self.x = self.F @ self.x
        self.x[0, 0] = self.normalize_angle(self.x[0, 0])

        # Propagate covariance through model uncertainty.
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measured_theta):
        """
        Perform the Kalman update step using a measured orientation angle.

        Update equations:
            y = normalized angular residual
            S = HPH^T + R
            K = PH^T S^-1
            x = x + Ky
            P = (I - KH)P

        Why residual normalization is important:
            Orientation differences near boundaries can appear large without
            wrapping; normalization keeps update corrections physically valid.

        Args:
            measured_theta (float): Measured orientation angle in degrees.

        Returns:
            np.ndarray: Updated state vector with shape (2, 1).
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling update().")

        measured_theta = self.normalize_angle(measured_theta)
        predicted_theta = float(self.x[0, 0])

        # Compute angular residual and normalize to avoid wrap-around artifacts.
        residual = measured_theta - predicted_theta
        residual = self.normalize_angle(residual)

        y = np.array([[residual]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Correct state with measurement-informed innovation.
        self.x = self.x + (K @ y)
        self.x[0, 0] = self.normalize_angle(self.x[0, 0])

        # Update covariance after correction.
        self.P = (self.I - (K @ self.H)) @ self.P
        return self.x

    def get_angle(self):
        """
        Return the current estimated orientation angle.

        Returns:
            float: Estimated angle in degrees normalized to [-90, 90].
        """
        return self.normalize_angle(self.x[0, 0])
