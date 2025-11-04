
import torch
import pytest
from dronecontrol.commande.optimizer import SimpleOptimizer


def simple_quadratic_loss(u, x_ref):
    return torch.mean((u - x_ref) ** 2)


def test_optimizer_initialization():
    optimizer = SimpleOptimizer(
        trajectory_loss_fn=simple_quadratic_loss,
        lr=1.0,
        max_iter=5,
        history_size=5,
        max_epochs=2,
    )

    assert callable(optimizer.trajectory_loss_fn)
    assert optimizer.lr == 1.0
    assert optimizer.max_iter == 5
    assert optimizer.history_size == 5
    assert optimizer.max_epochs == 2


def test_optimization_converges_quadratic():
    # small problem so LBFGS converges quickly
    target = torch.randn(8, 1)
    u_init = torch.zeros_like(target)

    optimizer = SimpleOptimizer(
        trajectory_loss_fn=simple_quadratic_loss,
        lr=1.0,
        max_iter=10,
        history_size=5,
        max_epochs=8,
    )

    u_opt = optimizer.optimize(u_init, target, verbose=False)

    initial_loss = simple_quadratic_loss(u_init, target).item()
    final_loss = simple_quadratic_loss(u_opt, target).item()

    assert final_loss < initial_loss
    assert u_opt.shape == target.shape


def test_get_tensions_bounds_and_values():
    tensions = torch.tensor([[0.0], [2.0], [-2.0]])
    u_min, u_max = -1.0, 1.0

    out = SimpleOptimizer.get_tensions(tensions, u_min, u_max)
    assert torch.all(out >= u_min - 1e-6)
    assert torch.all(out <= u_max + 1e-6)

    # check monotonicity
    assert out[1] > out[0] > out[2]


def test_optimization_with_constraints_improves_loss():
    target = torch.tensor([[0.2], [0.4], [-0.3]])
    u_init = torch.zeros_like(target)
    u_min, u_max = -1.0, 1.0

    def constrained_loss(u, x_ref):
        u_actual = SimpleOptimizer.get_tensions(u, u_min, u_max)
        return torch.mean((u_actual - x_ref) ** 2)

    optimizer = SimpleOptimizer(
        trajectory_loss_fn=constrained_loss,
        lr=0.5,
        max_iter=8,
        history_size=4,
        max_epochs=8,
    )

    u_opt_tensions = optimizer.optimize(u_init, target, verbose=False)
    final_loss = constrained_loss(u_opt_tensions, target).item()
    initial_loss = constrained_loss(u_init, target).item()
    assert final_loss < initial_loss


def test_zero_max_epochs_returns_initial():
    target = torch.randn(4, 1)
    u_init = torch.zeros_like(target)

    optimizer = SimpleOptimizer(
        trajectory_loss_fn=simple_quadratic_loss,
        lr=1.0,
        max_iter=5,
        history_size=3,
        max_epochs=0,
    )

    result = optimizer.optimize(u_init, target, verbose=False)
    # if no epochs, optimizer should return u_init (detached)
    assert torch.allclose(result, u_init)


def test_verbose_prints(capsys):
    target = torch.randn(6, 1)
    u_init = torch.zeros_like(target)

    optimizer = SimpleOptimizer(
        trajectory_loss_fn=simple_quadratic_loss,
        lr=1.0,
        max_iter=3,
        history_size=3,
        max_epochs=5,
    )

    optimizer.optimize(u_init, target, verbose=True)
    captured = capsys.readouterr()
    assert "Optimization completed" in captured.out


def test_hover_control_constant_due_to_gravity():
    """Use the existing TrajectoryLoss: when x_ref == x0 the optimizer
    should produce a nearly-constant thrust (due to gravity offset).
    """
    import torch.nn as nn
    from dronecontrol.commande.loss import TrajectoryLoss

    g = 9.81
    dt = 0.1
    horizon = 20

    x0_pos = 1.0
    x0_vel = 0.0

    # reference equal to initial position across horizon
    x_ref = torch.full((horizon, 1), x0_pos)

    # initial control guess
    u_init = torch.zeros_like(x_ref)

    # accel_model maps control u to acceleration; include gravity offset
    class AccelModel(nn.Module):
        def __init__(self, g_val: float):
            super().__init__()
            self.g = g_val

        def forward(self, u_tensor: torch.Tensor) -> torch.Tensor:
            return u_tensor - self.g

    accel_model = AccelModel(g)

    # simple quadratic Q and R (identity)
    Q = torch.eye(horizon)
    R = torch.eye(horizon)

    loss_module = TrajectoryLoss(accel_model=accel_model, horizon=horizon, dt=dt, Q_tensor=Q, R_tensor=R)

    # wrapper to match SimpleOptimizer signature (u, x_ref)
    def traj_loss_wrapper(u, x_ref_local):
        # TrajectoryLoss expects (u, x_ref, v0, x0)
        return loss_module(u, x_ref_local, torch.tensor([x0_vel]), torch.tensor([x0_pos]))

    optimizer = SimpleOptimizer(
        trajectory_loss_fn=traj_loss_wrapper,
        lr=1.0,
        max_iter=40,
        history_size=10,
        max_epochs=80,
    )

    u_opt = optimizer.optimize(u_init, x_ref, verbose=False)

    u_opt_np = u_opt.detach().cpu().numpy().flatten()
    mean_u = float(u_opt_np.mean())
    std_u = float(u_opt_np.std())

    # mean should be close to g (thrust balancing gravity) and fairly constant
    assert abs(mean_u - g) < 1.0, f"mean thrust {mean_u} not close to g={g}"
    assert std_u < 1.0, f"thrust not constant enough (std={std_u})"

if __name__ == '__main__':
    pytest.main([__file__])
