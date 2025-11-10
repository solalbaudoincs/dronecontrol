import torch
from typing import Tuple


class TrajectoryOptimizer:

    def __init__(
            self,
            dt: float,
            max_accel: float,
            x_ref: torch.Tensor,
            x0: float,
            smoothing: bool = True,
            alpha: float = 0.1,
            stable_time: float = 4,
            ):
        
        self.dt = dt
        self.max_accel = max_accel
        self.x_ref = x_ref
        self.x0 = x0
        self.smoothing = smoothing
        self.alpha = alpha
        self.nb_stable_steps = int(stable_time / dt)

    def _get_nb_steps(self, x, x0) -> int:
        """Compute number of steps from horizon and dt."""
        return int(abs(x - x0) / (self.max_accel * self.dt)) + 1

    @staticmethod
    def exponential_moving_average(x: torch.Tensor, alpha: float) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y


    def optimize_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize trajectory through all reference points in x_ref.
        Creates a trajectory from x0 -> x_ref[0] -> x_ref[1] -> ... -> x_ref[-1]
        """
        steps = []
        current_pos = self.x0
        
        # Iterate through each reference point
        for i in range(self.x_ref.shape[0]):
            x_target = self.x_ref[i].item()
            nb_steps = self._get_nb_steps(x_target, current_pos) * 3
            # Create trajectory segment from current position to target
            segment = torch.ones((nb_steps + self.nb_stable_steps), dtype=torch.float32)*x_target

            steps.append(segment)
            # Update current position for next segment
            current_pos = x_target
        
        if self.smoothing:
            # Concatenate all trajectory segments
            full_trajectory = torch.cat(steps)
            # Apply exponential moving average for smoothing
            full_trajectory = self.exponential_moving_average(full_trajectory, self.alpha)
        else :
            full_trajectory = torch.cat(steps)

        x_ref_step = torch.cat(steps)

        return full_trajectory, x_ref_step