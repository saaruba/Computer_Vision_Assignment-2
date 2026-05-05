import numpy as np


class TranslationKalmanFilter:
    """
    Kalman filter for 2D translation tracking with constant velocity dynamics.

    State vector:
        [x, y, vx, vy]^T

    Measurement vector:
        [x, y]^T
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=5.0):
        """
        Initialize Kalman filter matrices and parameters.

        Args:
            dt (float): Time step between frames.
            process_noise (float): Process noise scale.
            measurement_noise (float): Measurement noise scale.
        """
        self.dt = float(dt)

        # State transition matrix for constant velocity model.
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Measurement matrix (observes x and y only).
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        # Process noise covariance matrix.
        self.Q = float(process_noise) * np.array(
            [
                [self.dt**4 / 4.0, 0.0, self.dt**3 / 2.0, 0.0],
                [0.0, self.dt**4 / 4.0, 0.0, self.dt**3 / 2.0],
                [self.dt**3 / 2.0, 0.0, self.dt**2, 0.0],
                [0.0, self.dt**3 / 2.0, 0.0, self.dt**2],
            ],
            dtype=float,
        )

        # Measurement noise covariance matrix.
        self.R = float(measurement_noise) * np.eye(2, dtype=float)

        # State covariance matrix.
        self.P = np.eye(4, dtype=float) * 1000.0

        # Identity matrix.
        self.I = np.eye(4, dtype=float)

        # State vector [x, y, vx, vy]^T.
        self.x = np.zeros((4, 1), dtype=float)
        self.initialized = False

    def initialize(self, initial_x, initial_y):
        """
        Initialize state from first position measurement.

        Args:
            initial_x (float): Initial x coordinate.
            initial_y (float): Initial y coordinate.
        """
        self.x = np.array(
            [[float(initial_x)], [float(initial_y)], [0.0], [0.0]],
            dtype=float,
        )
        self.initialized = True

    def predict(self):
        """
        Run prediction step.

        Prediction:
            x = F x
            P = F P F^T + Q

        Returns:
            np.ndarray: Predicted state vector shape (4, 1).
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling predict().")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement_x, measurement_y):
        """
        Run measurement update step.

        Update:
            y = z - H x
            S = H P H^T + R
            K = P H^T inv(S)
            x = x + K y
            P = (I - K H) P

        Args:
            measurement_x (float): Measured x coordinate.
            measurement_y (float): Measured y coordinate.

        Returns:
            np.ndarray: Updated state vector shape (4, 1).
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling update().")

        z = np.array([[float(measurement_x)], [float(measurement_y)]], dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (self.I - (K @ self.H)) @ self.P
        return self.x

    def get_position(self):
        """
        Get current estimated position.

        Returns:
            tuple: (x, y) current estimated coordinates.
        """
        return float(self.x[0, 0]), float(self.x[1, 0])
