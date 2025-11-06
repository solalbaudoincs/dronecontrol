"""Profiler for `run_optim_report.py` that captures both CPU and GPU activity.

This script runs a shortened version of the report with reduced steps/epochs
and collects:
 - a Python cProfile binary (`.prof`) for CPU-level profiling,
 - a torch.profiler trace (Chrome trace JSON) for CUDA activity,
 - a text summary showing top CUDA/CPU hotspots.

Run with: `python run_optim_profile.py`
"""
import cProfile
import pstats
import io
from pathlib import Path
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from run_optim_report import ScenarioConfig, load_model, build_mpc, build_reference


def run_combined_profile(out_prof: Path, out_trace: Path, out_txt: Path):
    # Build config with reduced sizes for quick profiling
    cfg = ScenarioConfig()
    cfg.nb_steps = 80
    cfg.max_epochs = 8
    cfg.horizon = 20
    cfg.lr = 0.05

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model and controller
    model = load_model(cfg, device)
    controller = build_mpc(model, cfg)

    x0 = 1.0
    v0 = 0.0
    x_ref = build_reference(cfg.nb_steps)

    # 1) CPU profiling (cProfile) for Python-level hotspots
    pr = cProfile.Profile()
    pr.enable()

    # 2) Torch profiler for CUDA activity (if available)
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    # Warm-up a small controller run to avoid first-call overhead
    _ = controller.solve(x_ref=x_ref, x0=x0, v0=v0, verbose=False)

    if use_cuda:
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("run_mpc_solve"):
            # Full short run (the main thing we want to inspect)
            u_hist, x_hist, v_hist, a_hist = controller.solve(
                x_ref=x_ref,
                x0=x0,
                v0=v0,
                verbose=False,
            )

    if use_cuda:
        torch.cuda.synchronize()

    pr.disable()

    # Save cProfile binary for Python-level inspection
    out_prof.parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(str(out_prof))

    # Save torch profiler chrome trace for viewing in Chrome/TensorBoard
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    try:
        prof.export_chrome_trace(str(out_trace))
    except Exception:
        # Older torch versions may not implement export_chrome_trace
        pass

    # Prepare text summary combining cProfile and torch profiler key_averages
    s = io.StringIO()
    s.write("=== cProfile (top 30 by cumulative time) ===\n")
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats("cumtime").print_stats(30)

    s.write("\n=== torch.profiler key_averages (top 50 by self_cuda_time_total) ===\n")
    try:
        ka = prof.key_averages()
        # CUDA-heavy summary
        s.write(ka.table(sort_by="self_cuda_time_total", row_limit=50))
        s.write("\n\n")
        s.write(ka.table(sort_by="self_cpu_time_total", row_limit=50))
    except Exception:
        s.write("(torch.profiler summary not available)\n")

    out_txt.write_text(s.getvalue(), encoding="utf-8")

    print(f"cProfile saved to: {out_prof}")
    print(f"Trace (chrome) saved to: {out_trace}")
    print(f"Combined summary written to: {out_txt}")


if __name__ == "__main__":
    out_prof = Path("run_optim_profile.prof")
    out_trace = Path("run_optim_torch_trace.json")
    out_txt = Path("run_optim_profile_summary.txt")
    run_combined_profile(out_prof, out_trace, out_txt)
