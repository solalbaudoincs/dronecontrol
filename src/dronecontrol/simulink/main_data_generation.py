# exécution recommandée : uv run -m dronecontrol.simulink.main_data_generation
import numpy as np
from dronecontrol.simulink.generate_data import *

from dronecontrol.simulink.simulator import DroneSimulator
from pathlib import Path
from tqdm import tqdm 

# repo_root = .../dronecontrol  (car __file__ = .../src/dronecontrol/simulink/main_data_generation.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTDIR = REPO_ROOT / "data" / "control_4_motors"
OUTDIR.mkdir(parents=True, exist_ok=True)



def PRBS(duration, Te, u_min, u_max, t_min=0, t_max=None):
    nb_data = int(duration/Te)
    if t_max is None:
        t_max = duration/5.0
    n_min = int(t_min/Te)
    n_max = int(t_max/Te)
    sig = np.zeros(nb_data)
    c = 0
    while c < nb_data:
        amp = np.random.uniform(u_min, u_max)
        dur = int(min(np.random.uniform(n_min, n_max), nb_data - c))
        sig[c:c+dur] = amp
        c += dur
    return sig


def main():
    # ----------------- paramètres généraux -----------------
    N_RUNS = 1000
    Te = 0.05
    duration = 100.0
    u_min, u_max = 0.0, 10.0
    t_min = 5 * Te
    t_max = 2.0
    x0 = np.zeros(12, dtype=float)
    sim = DroneSimulator(initial_state=np.zeros(12))

    # Dossier de sortie demandé: dronecontrol/data/control_4_motors/
    REPO_ROOT = Path(__file__).resolve().parents[3]
    OUTDIR = REPO_ROOT / "data" / "control_4_motors"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ----------------- accumulateurs -----------------
    all_states = []       # liste de tableaux (Ni+1, 12) par run
    all_derivs = []       # liste de tableaux (Ni, 12) par run
    all_inputs = []       # liste de tableaux (Ni, 4) par run
    all_states_id = []    # run_id aligné sur lignes de states
    all_derivs_id = []    # run_id aligné sur lignes de derivs
    all_inputs_id = []    # run_id aligné sur lignes d'inputs

    # ----------------- boucle sur les runs -----------------
    for run_id in tqdm(range(N_RUNS)):
        # génère un PRBS différent à chaque run
        prbs1 = PRBS(duration, Te, u_min, u_max, t_min, t_max)
        prbs2 = PRBS(duration, Te, u_min, u_max, t_min, t_max)
        prbs3 = PRBS(duration, Te, u_min, u_max, t_min, t_max)
        prbs4 = PRBS(duration, Te, u_min, u_max, t_min, t_max)

        U = np.column_stack([prbs1, prbs2, prbs3, prbs4])      # (N,4)
        u_t = [U[k, :].astype(float) for k in range(U.shape[0])]

        # on ne sauvegarde pas à chaque run: save_to=None
        history, deriv_history = generate_data(sim, u_t, x0, save_to=None)

        states = np.vstack(history)          # (N+1, 12)
        derivs = np.vstack(deriv_history)    # (N, 12)
        inputs = U                            # (N, 4)

        all_states.append(states)
        all_derivs.append(derivs)
        all_inputs.append(inputs)

        all_states_id.append(np.full((states.shape[0], 1), run_id, dtype=int))
        all_derivs_id.append(np.full((derivs.shape[0], 1), run_id, dtype=int))
        all_inputs_id.append(np.full((inputs.shape[0], 1), run_id, dtype=int))

    # ----------------- concaténation + sauvegarde -----------------
    # Ajoute une première colonne run_id pour séparer les runs dans un seul CSV

    ST = np.hstack([np.vstack(all_states_id), np.vstack(all_states)])       # ((sum(Ni)+N_RUNS), 1+12)
    DV = np.hstack([np.vstack(all_derivs_id), np.vstack(all_derivs)])       # ((sum(Ni)), 1+12)
    IN = np.hstack([np.vstack(all_inputs_id), np.vstack(all_inputs)])       # ((sum(Ni)), 1+4)

    # noms de colonnes (en commentaire header)
    states_header = "run_id," + ",".join([f"x{i}" for i in range(12)])
    derivs_header = "run_id," + ",".join([f"dx{i}" for i in range(12)])
    inputs_header = "run_id,u1,u2,u3,u4"

    # fichiers de sortie (toujours les mêmes noms)
    np.savetxt(OUTDIR / "_states.csv",      ST, delimiter=",", fmt="%.10g", header=states_header, comments="")
    np.savetxt(OUTDIR / "_derivatives.csv", DV, delimiter=",", fmt="%.10g", header=derivs_header, comments="")
    np.savetxt(OUTDIR / "_inputs.csv",      IN, delimiter=",", fmt="%.10g", header=inputs_header, comments="")

    print(f"✅ Sauvegardé {N_RUNS} runs concaténés dans : {OUTDIR}")


def main2(run_id: int = 0, dt: float = 0.05, save: bool = False):
    """
    Charge les CSV concaténés dans OUTDIR et trace uniquement le graphe des angles (θ, φ) pour le run choisi.
    """
    save_path = OUTDIR if save else None
    print(f"▶️  Plot angles pour run_id={run_id} depuis {OUTDIR} (dt={dt})")
    # plot_run_from_csv vient de generate_data.py (importée via *)
    plot_run_from_csv(OUTDIR.as_posix(), run_id=run_id, dt=dt, show=True, save_path=save_path)

if __name__ == "__main__":
    # Choisis l’un des deux :
    main()   # pour générer + enregistrer les 10 runs concaténés
