import torch

class TrajectoryOptimizer:

    def __init__(
            self,
            dt: float,
            max_speed: float,
            x_ref: torch.Tensor,
            x0: float
            ):
        self.dt = dt
        self.max_speed = max_speed
        self.x_ref = x_ref
        self.x0 = x0

    def _get_nb_steps(self, x, x0) -> int:
        """Compute number of steps from horizon and dt."""
        return int(abs(x - x0) / (self.max_speed * self.dt)) + 1

    @staticmethod
    def exponential_moving_average(x: torch.Tensor, alpha: float) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[0] = x[0]
        y[:-1] = alpha * x[:-1] + (1 - alpha) * y[1:]
        return y


    def optimize_trajectory(self) -> torch.Tensor:
        """
        Optimize trajectory through all reference points in x_ref.
        Creates a trajectory from x0 -> x_ref[0] -> x_ref[1] -> ... -> x_ref[-1]
        """
        trajectories = []
        current_pos = self.x0
        
        # Iterate through each reference point
        for i in range(self.x_ref.shape[0]):
            x_target = self.x_ref[i].item()
            nb_steps = self._get_nb_steps(x_target, current_pos)
            # Create trajectory segment from current position to target
            segment = torch.ones((nb_steps), dtype=torch.float32)
            segment[0] = current_pos
            segment[1:] *= x_target
            smoothed_segment = self.exponential_moving_average(segment, alpha=0.1)
            trajectories.append(smoothed_segment)
            # Update current position for next segment
            current_pos = x_target
        
        # Concatenate all trajectory segments
        full_trajectory = torch.cat(trajectories)

        return full_trajectory