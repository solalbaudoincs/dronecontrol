from typing import Optional

from dronecontrol.utils import ensure_dir
from simulator import DroneSimulator, DT
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path


logger = logging.getLogger(__name__)



def generate_data(sim : DroneSimulator,u_t : list[np.ndarray], x0 : np.ndarray, save_to : Optional[str]) -> tuple[np.ndarray]:
    """
    Generate simulation data using the DroneSimulator.
    
    erxample usage:
    u_t = [np.array([0.0, 0.0, 0.0, 9.81])] * 200  # hover input for 200 time steps
    x0 = np.zeros(12)  # initial state at rest on the ground    
    history, deriv_history = generate_data(u_t, x0, save_to="simulation_output_path")
    """
    assert u_t[0].shape == (4,) and x0.shape == (12,)


    sim.reset(initial_state=x0)

    logger.info("=== Starting data generation using DroneSimulator ===")
    total_time = len(u_t) * DT
    logger.info(f"Total simulation time: {total_time:.2f} seconds ({len(u_t)} steps at DT={DT}s)")

    _, trace = sim.rollout(u_t, dt=DT, return_trace=True)  # Warm-up to set initial state

    dx_roll = trace["DX"]
    X_roll = trace["X"]

    history = np.array(X_roll)
    deriv_history = np.array(dx_roll)
    # for t in range(len(u_t)):
    #     st, dxdt = sim.step(u_t[t])
    #     history.append(np.array(sim.state))
    #     deriv_history.append(np.array(dxdt))

    
    logger.info("=== Data generation completed ===")
    if save_to is not None:
        logger.info(f"Saving generated data to {save_to}")
        ensure_dir(save_to)
        np.savetxt(save_to + '/_states.csv', np.array(X_roll), delimiter=',')
        np.savetxt(save_to + '/_derivatives.csv', np.array(dx_roll), delimiter=',')
        np.savetxt(save_to + '/_inputs.csv', np.array(u_t), delimiter=',')
        logger.info("Data saved successfully.")
    return history, deriv_history 


def load_simulation_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load simulation data from CSV files."""
    states = np.loadtxt(path + '/_states.csv', delimiter=',')
    derivatives = np.loadtxt(path + '/_derivatives.csv', delimiter=',')
    inputs = np.loadtxt(path + '/_inputs.csv', delimiter=',')
    return states, derivatives, inputs





def plot_run_from_csv(path: str, run_id: int, dt: float = 0.05, show: bool = True, save_path: str | None = None):
    """
    Trace les courbes du run spécifié (run_id) à partir des CSV concaténés :
      - _states.csv
      - _derivatives.csv
      - _inputs.csv

    Paramètres
    ----------
    path : dossier contenant les fichiers CSV
    run_id : identifiant du run à tracer (0, 1, 2, ...)
    dt : pas d'échantillonnage en secondes
    show : affiche les figures si True
    save_path : si non None, enregistre les images dans ce dossier
    """
    path = Path(path)
    states_path = path / "_states.csv"
    derivs_path = path / "_derivatives.csv"
    inputs_path = path / "_inputs.csv"

    # --- Lecture CSV ---
    states_all = np.loadtxt(states_path, delimiter=",", skiprows=1)
    derivs_all = np.loadtxt(derivs_path, delimiter=",", skiprows=1)
    inputs_all = np.loadtxt(inputs_path, delimiter=",", skiprows=1)

    # --- Sélection du run ---
    states = states_all[states_all[:, 0] == run_id, 1:]     # (N+1, 12)
    derivs = derivs_all[derivs_all[:, 0] == run_id, 1:]     # (N, 12)
    inputs = inputs_all[inputs_all[:, 0] == run_id, 1:]     # (N, 4)

    if states.size == 0:
        raise ValueError(f"Aucun run_id={run_id} trouvé dans {path}")

    N = inputs.shape[0]
    t_u = np.arange(N) * dt
    t_x = np.arange(N + 1) * dt
    t_a = np.arange(N) * dt

    # ================= Fig 1 : Entrées u₁..u₄ =================
    fig_u = plt.figure(figsize=(10, 6))
    for k in range(4):
        plt.plot(t_u, inputs[:, k], label=f"u{k+1}")
    plt.xlabel("Temps (s)")
    plt.ylabel("Commande (u)")
    plt.title(f"Run {run_id} — Entrées moteur (u₁..u₄)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    # ================= Fig 2 : Positions x, y, z =================
    fig_pos = plt.figure(figsize=(10, 5))
    plt.plot(t_x, states[:, 0], label="x")
    plt.plot(t_x, states[:, 1], label="y")
    plt.plot(t_x, states[:, 2], label="z")
    plt.xlabel("Temps (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Run {run_id} — Positions (x, y, z)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    # ================= Fig 3 : Accélérations aₓ, a_y, a_z =================
    fig_acc = plt.figure(figsize=(10, 5))
    plt.plot(t_a, derivs[:, 3], label="aₓ")
    plt.plot(t_a, derivs[:, 4], label="a_y")
    plt.plot(t_a, derivs[:, 5], label="a_z")
    plt.xlabel("Temps (s)")
    plt.ylabel("Accélération (m/s²)")
    plt.title(f"Run {run_id} — Accélérations (aₓ, a_y, a_z)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    # ================= Fig 4 : Angles θ, φ uniquement =================
    # on convertit rad→deg et wrap dans [-180,180]
    angles = np.rad2deg(states[:, 6:8].copy())  # colonnes θ, φ uniquement
    angles = (angles + 180.0) % 360.0 - 180.0
    theta = angles[:, 0]  # pitch
    phi   = angles[:, 1]  # roll

    # zones "drone à l’envers"
    inverted = (np.abs(theta) > 90.0) | (np.abs(phi) > 90.0)

    fig_ang = plt.figure(figsize=(10, 5))
    ax = fig_ang.gca()
    ax.plot(t_x, theta, label="θ (pitch)", linewidth=1.8)
    ax.plot(t_x, phi, label="φ (roll)", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Angle (°)")
    ax.set_ylim([-180, 180])
    ax.set_title(f"Run {run_id} — Angles (θ, φ) wrap [-180°, 180°]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # colorie les zones où le drone est à l’envers
    if inverted.any():
        run_on = False
        start_idx = 0
        for k, flag in enumerate(inverted):
            if flag and not run_on:
                run_on = True
                start_idx = k
            elif run_on and (not flag or k == len(inverted) - 1):
                end_idx = k if not flag else k
                ax.fill_between(t_x[start_idx:end_idx + 1], -180, 180, alpha=0.12, step="pre")
                run_on = False

    # --- Sauvegarde éventuelle ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig_u.savefig(save_path / f"run{run_id}_inputs.png", dpi=150, bbox_inches="tight")
        fig_pos.savefig(save_path / f"run{run_id}_positions.png", dpi=150, bbox_inches="tight")
        fig_acc.savefig(save_path / f"run{run_id}_accelerations.png", dpi=150, bbox_inches="tight")
        fig_ang.savefig(save_path / f"run{run_id}_angles.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return {
        "fig_u": fig_u,
        "fig_pos": fig_pos,
        "fig_acc": fig_acc,
        "fig_ang": fig_ang,
    }
