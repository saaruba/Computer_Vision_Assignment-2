import numpy as np


class RotationKalmanFilter:
    """
    Kalman filter for orientation tracking using a constant angular velocity model.

    State vector:
        [theta, omega]^T
        theta: angle in degrees
        omega: angular velocity in degrees/frame

    Measurement vector:
        [theta]^T
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

        # State transition matrix for:
        # theta_t = theta_(t-1) + omega_(t-1) * dt
        # omega_t = omega_(t-1)
        self.F = np.array(
            [
                [1.0, self.dt],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        # Measurement matrix (we only measure theta).
        self.H = np.array([[1.0, 0.0]], dtype=float)

        # Process noise covariance (constant angular velocity model).
        self.Q = float(process_noise) * np.array(
            [
                [self.dt**4 / 4.0, self.dt**3 / 2.0],
                [self.dt**3 / 2.0, self.dt**2],
            ],
            dtype=float,
        )

        # Measurement noise covariance.
        self.R = np.array([[float(measurement_noise)]], dtype=float)

        # State covariance matrix.
        self.P = np.eye(2, dtype=float) * 1000.0

        # Identity matrix.
        self.I = np.eye(2, dtype=float)

        # State vector [theta, omega]^T.
        self.x = np.zeros((2, 1), dtype=float)

        self.initialized = False

    def normalize_angle(self, angle):
        """
        Normalize angle to [-90, 90] degrees.

        Args:
            angle (float): Input angle in degrees.

        Returns:
            float: Normalized angle in [-90, 90].
        """
        return ((float(angle) + 90.0) % 180.0) - 90.0

    def initialize(self, initial_theta):
        """
        Initialize the filter with first angle measurement.

        Args:
            initial_theta (float): Initial angle in degrees.
        """
        theta0 = self.normalize_angle(initial_theta)
        self.x = np.array([[theta0], [0.0]], dtype=float)
        self.initialized = True

    def predict(self):
        """
        Perform Kalman prediction step.

        Equations:
            x = F x
            P = F P F.T + Q

        Returns:
            np.ndarray: Predicted state vector shape (2, 1).
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling predict().")

        self.x = self.F @ self.x
        self.x[0, 0] = self.normalize_angle(self.x[0, 0])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measured_theta):
        """
        Perform Kalman update step using measured angle.

        Equations:
            y = normalized angular residual
            S = H P H.T + R
            K = P H.T inv(S)
            x = x + K y
            P = (I - K H) P

        Args:
            measured_theta (float): Measured angle in degrees.

        Returns:
            np.ndarray: Updated state vector shape (2, 1).
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before calling update().")

        measured_theta = self.normalize_angle(measured_theta)
        predicted_theta = float(self.x[0, 0])

        residual = measured_theta - predicted_theta
        residual = self.normalize_angle(residual)

        y = np.array([[residual]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.x[0, 0] = self.normalize_angle(self.x[0, 0])
        self.P = (self.I - (K @ self.H)) @ self.P

        return self.x

    def get_angle(self):
        """
        Get current estimated angle in degrees.

        Returns:
            float: Current estimated angle normalized to [-90, 90].
        """
        return self.normalize_angle(self.x[0, 0])
