import os
import json
import csv
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.cost2100 import COST2100Dataset
from src.models.lite_dpas import LiteDPASCsiNetV5, count_parameters
from src.profile_lite_dpas import approximate_macs
from src.metrics import sample_nmse_linear

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent
    runs_dir = project_root / "runs"
    source_dir = project_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(r"C:\Users\신동희\Desktop\CSI\COST2100")
    if not data_dir.exists():
        data_dir = project_root / "Data"

    runs_info = [
        ("indoor", 4, "litedpas_v5_indoor_cr4_1000ep"),
        ("indoor", 16, "litedpas_v5_indoor_cr16_1000ep"),
        ("indoor", 32, "litedpas_v5_indoor_cr32_1000ep"),
        ("indoor", 64, "litedpas_v5_indoor_cr64_1000ep"),
        ("outdoor", 4, "litedpas_v5_outdoor_cr4_1000ep"),
        ("outdoor", 16, "litedpas_v5_outdoor_cr16_1000ep"),
        ("outdoor", 32, "litedpas_v5_outdoor_cr32_1000ep"),
        ("outdoor", 64, "litedpas_v5_outdoor_cr64_1000ep"),
    ]

    def calc_cosine_similarity(gt, pred):
        g = (gt - 0.5).reshape(gt.shape[0], -1)
        p = (pred - 0.5).reshape(pred.shape[0], -1)
        dot = (g * p).sum(dim=1)
        norm_g = torch.norm(g, dim=1)
        norm_p = torch.norm(p, dim=1)
        cos = dot / (norm_g * norm_p + 1e-8)
        return cos.mean().item()

    test_loaders = {}
    val_loaders = {}
    for scenario_type, s_suffix in [("indoor", "in"), ("outdoor", "out")]:
        t_path = data_dir / f"DATA_Htest{s_suffix}.mat"
        if t_path.exists():
            test_loaders[scenario_type] = DataLoader(
                COST2100Dataset(t_path), batch_size=200, shuffle=False
            )
        v_path = data_dir / f"DATA_Hval{s_suffix}.mat"
        if v_path.exists():
            val_loaders[scenario_type] = DataLoader(
                COST2100Dataset(v_path), batch_size=200, shuffle=False
            )

    results = []
    curves_data = {"indoor": {}, "outdoor": {}}

    for scenario, cr, rname in runs_info:
        rdir = runs_dir / rname
        history_file = rdir / "history.csv"
        best_ckpt = rdir / "best.pt"

        # Read history for plotting
        if history_file.is_file():
            epochs, val_nmse_dbs, train_mses = [], [], []
            with open(history_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    val_nmse_dbs.append(float(row["val_nmse_db"]))
                    train_mses.append(float(row["train_mse"]))
            curves_data[scenario][cr] = {
                "epochs": epochs,
                "val_nmse_db": val_nmse_dbs,
                "train_mse": train_mses,
            }

        if not best_ckpt.is_file():
            print(f"Warning: {best_ckpt} not found!")
            continue

        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        encoded_dim = ckpt["encoded_dim"]
        model = LiteDPASCsiNetV5(encoded_dim=encoded_dim)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()

        params = count_parameters(model)
        dummy = torch.zeros(1, 2, 32, 32, device=device)
        macs = approximate_macs(model, dummy)

        # Eval on Val Set
        val_nmse_list, val_rho_list = [], []
        if scenario in val_loaders:
            with torch.no_grad():
                for x in val_loaders[scenario]:
                    x = x.to(device)
                    out = model(x)
                    val_nmse_list.append(sample_nmse_linear(x, out).cpu())
                    val_rho_list.append(calc_cosine_similarity(x, out))
            val_linear_mean = torch.cat(val_nmse_list).mean().item()
            val_nmse_db = 10.0 * np.log10(max(val_linear_mean, 1e-12))
            val_rho = float(np.mean(val_rho_list))
        else:
            val_nmse_db = ckpt.get("val_nmse_db", 0.0)
            val_rho = 0.0

        # Eval on Test Set
        test_nmse_list, test_rho_list = [], []
        if scenario in test_loaders:
            with torch.no_grad():
                for x in test_loaders[scenario]:
                    x = x.to(device)
                    out = model(x)
                    test_nmse_list.append(sample_nmse_linear(x, out).cpu())
                    test_rho_list.append(calc_cosine_similarity(x, out))
            test_linear_mean = torch.cat(test_nmse_list).mean().item()
            test_nmse_db = 10.0 * np.log10(max(test_linear_mean, 1e-12))
            test_rho = float(np.mean(test_rho_list))
        else:
            test_nmse_db = 0.0
            test_rho = 0.0

        res = {
            "scenario": scenario,
            "compression_ratio": f"1/{cr}",
            "cr_int": cr,
            "encoded_dim": encoded_dim,
            "best_epoch": int(ckpt.get("epoch", -1)),
            "val_nmse_db": round(val_nmse_db, 3),
            "val_rho": round(val_rho, 5),
            "test_nmse_db": round(test_nmse_db, 3),
            "test_rho": round(test_rho, 5),
            "parameters": params,
            "macs": macs,
            "flops": macs * 2,
        }
        results.append(res)
        print(f"[{scenario.capitalize()} CR=1/{cr}] Val NMSE: {res['val_nmse_db']:.3f} dB (rho={res['val_rho']:.5f}) | Test NMSE: {res['test_nmse_db']:.3f} dB (rho={res['test_rho']:.5f}) | Params: {params:,} | MACs: {macs:,}")

    with open(source_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plotting styles
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {4: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c", 64: "#d62728"}

    # 1. Indoor Validation NMSE Curve
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for cr in [4, 16, 32, 64]:
        if cr in curves_data["indoor"]:
            data = curves_data["indoor"][cr]
            ax.plot(data["epochs"], data["val_nmse_db"], label=f"CR = 1/{cr}", color=colors[cr], linewidth=1.8)
    ax.set_title("Indoor Scenario - Validation NMSE Convergence (1000 Epochs)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12, labelpad=8)
    ax.set_ylabel("Validation NMSE (dB)", fontsize=12, labelpad=8)
    ax.legend(title="Compression Ratio", fontsize=11, title_fontsize=11, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(source_dir / "chart_indoor.png", dpi=300)
    plt.close()

    # 2. Outdoor Validation NMSE Curve
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for cr in [4, 16, 32, 64]:
        if cr in curves_data["outdoor"]:
            data = curves_data["outdoor"][cr]
            ax.plot(data["epochs"], data["val_nmse_db"], label=f"CR = 1/{cr}", color=colors[cr], linewidth=1.8)
    ax.set_title("Outdoor Scenario - Validation NMSE Convergence (1000 Epochs)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12, labelpad=8)
    ax.set_ylabel("Validation NMSE (dB)", fontsize=12, labelpad=8)
    ax.legend(title="Compression Ratio", fontsize=11, title_fontsize=11, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(source_dir / "chart_outdoor.png", dpi=300)
    plt.close()

    # 3. Side-by-side comparison (Indoor & Outdoor)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    for cr in [4, 16, 32, 64]:
        if cr in curves_data["indoor"]:
            data = curves_data["indoor"][cr]
            ax1.plot(data["epochs"], data["val_nmse_db"], label=f"CR = 1/{cr}", color=colors[cr], linewidth=1.8)
    ax1.set_title("Indoor Scenario", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation NMSE (dB)", fontsize=11)
    ax1.legend(title="Compression Ratio", fontsize=10, title_fontsize=10, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    for cr in [4, 16, 32, 64]:
        if cr in curves_data["outdoor"]:
            data = curves_data["outdoor"][cr]
            ax2.plot(data["epochs"], data["val_nmse_db"], label=f"CR = 1/{cr}", color=colors[cr], linewidth=1.8)
    ax2.set_title("Outdoor Scenario", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation NMSE (dB)", fontsize=11)
    ax2.legend(title="Compression Ratio", fontsize=10, title_fontsize=10, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Validation NMSE Convergence Curves (1000 Epochs)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(source_dir / "chart_convergence_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Bar chart comparing NMSE & Cosine Sim across CRs
    indoor_res = [r for r in results if r["scenario"] == "indoor"]
    outdoor_res = [r for r in results if r["scenario"] == "outdoor"]
    
    crs = [4, 16, 32, 64]
    x = np.arange(len(crs))
    width = 0.35

    # Test NMSE Bar chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    in_nmses = [next(r["test_nmse_db"] for r in indoor_res if r["cr_int"] == c) for c in crs]
    out_nmses = [next(r["test_nmse_db"] for r in outdoor_res if r["cr_int"] == c) for c in crs]
    
    rects1 = ax.bar(x - width/2, in_nmses, width, label="Indoor", color="#3498db")
    rects2 = ax.bar(x + width/2, out_nmses, width, label="Outdoor", color="#e74c3c")
    ax.set_ylabel("Test NMSE (dB)", fontsize=12)
    ax.set_title("Test NMSE Performance across Compression Ratios", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"1/{c}" for c in crs], fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}dB", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}dB", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, -12), textcoords="offset points", ha="center", va="top", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(source_dir / "chart_test_nmse_bars.png", dpi=300)
    plt.close()

    print("All plots and evaluation results saved to source/ directory successfully!")

if __name__ == "__main__":
    main()
