# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Context-Sensitivity ("Context-Flip") Experiment  --  NATURAL multi-context only
===============================================================================
Answers the question flagged in the paper (main.tex lines 425, 723):

  "Select promoter sequences appearing under multiple biological contexts in the
   test set, hold the sequence fixed, and record whether the predicted
   regulation direction changes."

Idea: some promoter sequences occur in the test set MORE THAN ONCE, each time
under a different biological context (a different variety / tissue / stage).
The promoter (its motif counts + sequence statistics) is byte-for-byte identical
across those rows -- only the biological context differs. So we can simply ask:
does the model predict a DIFFERENT regulation direction (up vs. down) for the
same promoter under its different real contexts?

If yes, the model is genuinely using biological context, not the promoter
sequence alone -- validating the context-conditional architecture. We also report
how often the ACTUAL (ground-truth) label changes across those same contexts, as
an upper bound / reality check.

Primary model: HybridGatedFusion (the proposed model). A per-model comparison of
the same natural flip rate is also produced.

Outputs (in ./results/):
  context_flip_report.txt            human-readable summary
  context_flip_summary.json          machine-readable summary
  natural_multicontext_examples.csv  illustrative flip cases (same seq, diff context)
  natural_flip_by_model.csv          per-model natural direction-change rate
  fig_context_flip.png/pdf           bar chart of natural flip rate per model

Usage:
    CUDA_VISIBLE_DEVICES=0 python context_flip_experiment.py
"""
import os
import sys
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2"
V3_DIR = os.path.join(PROJECT_DIR, "global motif counting based/output_global/hybrid_models_v3_reproducible_seed42")
MODELS_DIR = os.path.join(V3_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "global motif counting based/output_global_data")
OUT_DIR = os.path.join(PROJECT_DIR, "context_sensitivity_experiment", "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Import model classes from the v2 module (as v3 does). v2 redirects stdout at
# import time, so import from a scratch dir and restore stdout afterwards.
# ---------------------------------------------------------------------------
sys.path.insert(0, PROJECT_DIR)
_SCRATCH = "/tmp/ctxflip_scratch"
os.makedirs(_SCRATCH, exist_ok=True)
_cwd, _stdout = os.getcwd(), sys.stdout
os.chdir(_SCRATCH)
import rice_hybrid_classifier_global_v2 as v2  # noqa: E402
sys.stdout = _stdout
os.chdir(_cwd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load config + column lists + scaler + test data
# ---------------------------------------------------------------------------
with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
    cfg = json.load(f)
MOTIF, SEQ = cfg["motif_cols"], cfg["seq_stat_cols"]
VAR, TIS, STG = cfg["variety_cols"], cfg["tissue_cols"], cfg["stage_cols"]
HD, DR = cfg["hidden_dim"], cfg["dropout"]
nM, nV, nT, nS, nQ = cfg["num_motifs"], cfg["num_varieties"], cfg["num_tissues"], cfg["num_stages"], cfg["num_seq_stats"]

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).reset_index(drop=True)
N = len(test_df)
print(f"Test set: {N} samples | device: {device}")


def build_model(name):
    m = {
        "HybridMLP": lambda: v2.HybridMLP(nM, nV, nT, nS, nQ, HD, DR),
        "HybridLSTM": lambda: v2.HybridLSTM(nM, nV, nT, nS, nQ, HD, DR),
        "HybridTransformer": lambda: v2.HybridTransformer(nM, nV, nT, nS, nQ, HD, 4, 3, DR),
        "HybridCNNAttention": lambda: v2.HybridCNNAttention(nM, nV, nT, nS, nQ, HD, DR),
        "HybridGatedFusion": lambda: v2.HybridGatedFusion(nM, nV, nT, nS, nQ, HD, DR),
        "CrossAttentionFusion": lambda: v2.CrossAttentionFusion(nM, nV, nT, nS, nQ, HD, DR),
        "HybridMultiScaleCNN": lambda: v2.HybridMultiScaleCNN(nM, nV, nT, nS, nQ, HD, DR),
    }[name]()
    m.load_state_dict(torch.load(os.path.join(MODELS_DIR, name + ".pt"), map_location=device))
    m.to(device).eval()
    return m


# Pre-scale the sequence features once (they never change across contexts).
motif_raw = test_df[MOTIF].values.astype(np.float32)
seq_raw = test_df[SEQ].values.astype(np.float32)
combined_scaled = scaler.transform(np.hstack([motif_raw, seq_raw]))
motif_scaled = combined_scaled[:, :nM].astype(np.float32)
seq_scaled = combined_scaled[:, nM:].astype(np.float32)
var_oh = test_df[VAR].values.astype(np.float32)
tis_oh = test_df[TIS].values.astype(np.float32)
stg_oh = test_df[STG].values.astype(np.float32)
var_idx, tis_idx, stg_idx = var_oh.argmax(1), tis_oh.argmax(1), stg_oh.argmax(1)


@torch.no_grad()
def predict(model, batch=8192):
    """Predicted P(up-regulated) for every test row, in its ORIGINAL context."""
    out = np.empty(N, dtype=np.float32)
    for i in range(0, N, batch):
        sl = slice(i, min(i + batch, N))
        p = model(
            torch.from_numpy(motif_scaled[sl]).to(device),
            torch.from_numpy(seq_scaled[sl]).to(device),
            torch.from_numpy(var_oh[sl]).to(device),
            torch.from_numpy(tis_oh[sl]).to(device),
            torch.from_numpy(stg_oh[sl]).to(device),
        )
        out[sl] = np.atleast_1d(p.cpu().numpy().astype(np.float32).ravel())
    return out


# ---------------------------------------------------------------------------
# Identify NATURAL multi-context promoters (model-independent).
# A promoter is keyed by its exact raw (motif counts + seq stats) vector; rows
# sharing a key are the same sequence. We keep keys that occur under >= 2
# distinct (variety, tissue, stage) contexts.
# ---------------------------------------------------------------------------
seq_key = np.array(["|".join(f"{x:.6g}" for x in row)
                    for row in np.hstack([motif_raw, seq_raw])])
ctx = list(zip(var_idx, tis_idx, stg_idx))
test_df["_seq_key"] = seq_key
test_df["_ctx"] = ctx

n_unique = test_df["_seq_key"].nunique()
groups = {}  # seq_key -> row indices, only for multi-context promoters
for key, g in test_df.groupby("_seq_key"):
    if g["_ctx"].nunique() >= 2:
        groups[key] = g.index.values
n_multi = len(groups)
n_label_flip = sum(int(test_df.loc[idx, "label"].nunique() > 1) for idx in groups.values())

print(f"Unique promoters: {n_unique} | multi-context: {n_multi} | "
      f"with actual label change: {n_label_flip}")

# ---------------------------------------------------------------------------
# Per-model: fraction of multi-context promoters whose PREDICTED direction
# changes across the real contexts they appear under.
# ---------------------------------------------------------------------------
MODEL_ORDER = ["HybridGatedFusion", "HybridLSTM", "HybridCNNAttention",
               "HybridMultiScaleCNN", "HybridMLP", "HybridTransformer",
               "CrossAttentionFusion"]
PRIMARY = "HybridGatedFusion"

lab = test_df["label"].values.astype(int)

# Split the multi-context promoters by whether their ACTUAL label changes:
#   variable  -> label genuinely differs across contexts (context truly matters)
#   constant  -> same label in every context (context should NOT change the answer)
var_groups = [idx for idx in groups.values() if len(set(lab[idx])) > 1]
const_groups = [idx for idx in groups.values() if len(set(lab[idx])) == 1]
n_var, n_const = len(var_groups), len(const_groups)
var_rows = np.concatenate(var_groups) if var_groups else np.array([], dtype=int)
const_rows = np.concatenate(const_groups) if const_groups else np.array([], dtype=int)

per_model = {}
model_dir = {}
primary_base = None
for name in MODEL_ORDER:
    print(f"  analyzing {name} ...")
    model = build_model(name)
    base = predict(model)
    base_dir = (base > 0.5).astype(int)
    model_dir[name] = base_dir

    n_pred_flip = sum(int(len(set(base_dir[idx])) > 1) for idx in groups.values())

    # --- Correctness of flipping ---
    # Variable-label promoters: does the model predict EVERY context correctly?
    correct_all = sum(int(all(base_dir[i] == lab[i] for i in idx)) for idx in var_groups)
    flipped_var = sum(int(len(set(base_dir[idx])) > 1) for idx in var_groups)
    missed_var = sum(int(len(set(base_dir[idx])) == 1) for idx in var_groups)  # ignored context
    acc_var_rows = float((base_dir[var_rows] == lab[var_rows]).mean()) if len(var_rows) else None
    # Constant-label promoters: does the model wrongly flip?
    spurious = sum(int(len(set(base_dir[idx])) > 1) for idx in const_groups)
    acc_const_rows = float((base_dir[const_rows] == lab[const_rows]).mean()) if len(const_rows) else None

    per_model[name] = {
        "n_pred_direction_change": int(n_pred_flip),
        "frac_pred_direction_change": round(n_pred_flip / n_multi, 4) if n_multi else None,
        # among variable-label promoters (n_var):
        "var_correct_all": int(correct_all),
        "var_correct_all_rate": round(correct_all / n_var, 4) if n_var else None,
        "var_flipped_rate": round(flipped_var / n_var, 4) if n_var else None,
        "var_missed_rate": round(missed_var / n_var, 4) if n_var else None,
        "var_row_accuracy": round(acc_var_rows, 4) if acc_var_rows is not None else None,
        # among constant-label promoters (n_const):
        "const_spurious_flip": int(spurious),
        "const_spurious_flip_rate": round(spurious / n_const, 4) if n_const else None,
        "const_row_accuracy": round(acc_const_rows, 4) if acc_const_rows is not None else None,
    }
    if name == PRIMARY:
        primary_base = base

# ---------------------------------------------------------------------------
# Illustrative examples (primary model): same sequence, different context,
# largest predicted-probability spread within the group.
# ---------------------------------------------------------------------------
base = primary_base
base_dir = (base > 0.5).astype(int)
ex_groups = sorted(groups.items(),
                   key=lambda kv: -(base[kv[1]].max() - base[kv[1]].min()))
ex_records = []
for gi, (key, idx) in enumerate(ex_groups[:20], 1):
    spread = float(base[idx].max() - base[idx].min())
    for j in idx:
        r = test_df.loc[j]
        ex_records.append({
            "example_id": gi,
            "prob_spread_in_group": round(spread, 4),
            "variety": VAR[int(r["_ctx"][0])].replace("Variety_", ""),
            "tissue": TIS[int(r["_ctx"][1])].replace("Tissue_", ""),
            "stage": STG[int(r["_ctx"][2])].replace("Stage_", ""),
            "actual_label": "Up" if r["label"] == 1 else "Down",
            "pred_prob": round(float(base[j]), 4),
            "pred_dir": "Up" if base_dir[j] == 1 else "Down",
        })
pd.DataFrame(ex_records).to_csv(os.path.join(OUT_DIR, "natural_multicontext_examples.csv"), index=False)

# Per-model CSV (flip rate + correctness of flipping)
pd.DataFrame([{
    "model": n,
    "n_multi_context_promoters": n_multi,
    "n_pred_direction_change": per_model[n]["n_pred_direction_change"],
    "frac_pred_direction_change": per_model[n]["frac_pred_direction_change"],
    "n_variable_label_promoters": n_var,
    "var_correct_all_rate": per_model[n]["var_correct_all_rate"],
    "var_flipped_rate": per_model[n]["var_flipped_rate"],
    "var_missed_rate": per_model[n]["var_missed_rate"],
    "var_row_accuracy": per_model[n]["var_row_accuracy"],
    "n_constant_label_promoters": n_const,
    "const_spurious_flip_rate": per_model[n]["const_spurious_flip_rate"],
    "const_row_accuracy": per_model[n]["const_row_accuracy"],
} for n in MODEL_ORDER]).to_csv(os.path.join(OUT_DIR, "natural_flip_by_model.csv"), index=False)

# ---------------------------------------------------------------------------
# Figure: per-model natural predicted-direction-change rate, with a reference
# line at the actual-label-change rate.
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 13, "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"]})
SET3 = list(plt.get_cmap("Set3").colors)  # 12 soft qualitative colors
fig, ax = plt.subplots(figsize=(8.5, 5))
short = [m.replace("Hybrid", "") for m in MODEL_ORDER]
vals = [per_model[m]["frac_pred_direction_change"] for m in MODEL_ORDER]
# Set3[5] (orange) highlights the primary model; Set3[4] (blue) for the rest.
colors = [SET3[5] if m == PRIMARY else SET3[4] for m in MODEL_ORDER]
bars = ax.bar(short, vals, color=colors, edgecolor="#555555", linewidth=0.8)
lbl_ref = n_label_flip / n_multi
ax.axhline(lbl_ref, color="#333333", ls="--", lw=1.6,
           label=f"Actual label change ({100*lbl_ref:.1f}%)")
for b, v in zip(bars, vals):
    ax.annotate(f"{100*v:.1f}%", (b.get_x() + b.get_width()/2, v),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11)
ax.set_ylabel("Promoters with predicted\ndirection change")
ax.set_ylim(0, max(vals + [lbl_ref]) * 1.2)
ax.set_xticklabels(short, rotation=30, ha="right")
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_context_flip.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_context_flip.pdf"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: correctness of flipping.
#   - GREEN: on VARIABLE-label promoters (context truly flips the outcome),
#     fraction the model gets right in EVERY context (correct flip). Higher=better.
#   - RED: on CONSTANT-label promoters (outcome should NOT change), fraction the
#     model wrongly flips (spurious flip). Lower=better.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(MODEL_ORDER)); w = 0.38
cor = [per_model[m]["var_correct_all_rate"] for m in MODEL_ORDER]
spu = [per_model[m]["const_spurious_flip_rate"] for m in MODEL_ORDER]
cor_cnt = [per_model[m]["var_correct_all"] for m in MODEL_ORDER]
spu_cnt = [per_model[m]["const_spurious_flip"] for m in MODEL_ORDER]
# Set3[6] (green) = correct/good, Set3[3] (salmon) = spurious/bad.
b1 = ax.bar(x - w/2, cor, w, color=SET3[6], edgecolor="#555555", linewidth=0.8,
            label=f"Correct flip  (variable-label, n={n_var})")
b2 = ax.bar(x + w/2, spu, w, color=SET3[3], edgecolor="#555555", linewidth=0.8,
            label=f"Spurious flip (constant-label, n={n_const})")
# Percentage above each bar (real counts are reported in the paper table).
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{100*b.get_height():.0f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

# Delta indicator: the gap between correct (green) and spurious (red). The larger
# and more positive Delta = correct - spurious, the better the model separates
# genuine context-driven flips from spurious ones.
for xi, c, s in zip(x, cor, spu):
    d = c - s
    ax.annotate("", xy=(xi, c), xytext=(xi, s),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4))
    col = "#1B7A2F" if d > 0 else "#C0392B"
    ax.text(xi, max(c, s) + 0.045, rf"$\Delta{{=}}{100*d:+.0f}$", ha="center", va="bottom",
            fontweight="bold", fontsize=11.5, color=col)

# Wrap the one long label onto two lines so the plot area gets more room.
short_wrap = [s.replace("CrossAttentionFusion", "CrossAttention\nFusion") for s in short]
ax.set_xticks(x); ax.set_xticklabels(short_wrap, rotation=30, ha="right")
ax.set_xlabel("Model", fontsize=14, labelpad=1)
ax.set_ylabel("Fraction of promoters")
ax.set_ylim(0, max(cor + spu) * 1.40)
ax.text(0.015, 0.985, r"$\Delta$ = Correct $-$ Spurious  (larger is better)",
        transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
        style="italic", color="#333333")
ax.legend(fontsize=10.5, loc="upper right"); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_context_flip_correctness.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_context_flip_correctness.pdf"), bbox_inches="tight")
plt.close()

# Remove stale outputs from the earlier (Experiment-B) version, if present.
for stale in ["counterfactual_by_model.csv", "fig_context_sensitivity.png", "fig_context_sensitivity.pdf"]:
    p = os.path.join(OUT_DIR, stale)
    if os.path.exists(p):
        os.remove(p)

# ---------------------------------------------------------------------------
# Save JSON + human-readable report
# ---------------------------------------------------------------------------
pg = per_model[PRIMARY]
summary = {
    "seed": SEED,
    "primary_model": PRIMARY,
    "n_test_samples": int(N),
    "n_unique_promoters": int(n_unique),
    "n_promoters_multi_context": int(n_multi),
    "n_multi_context_with_actual_label_change": int(n_label_flip),
    "frac_multi_context_with_actual_label_change": round(n_label_flip / n_multi, 4) if n_multi else None,
    "n_variable_label_promoters": int(n_var),
    "n_constant_label_promoters": int(n_const),
    "per_model": per_model,
}
with open(os.path.join(OUT_DIR, "context_flip_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

L = []
A = L.append
A("=" * 88)
A("CONTEXT-FLIP EXPERIMENT (natural multi-context promoters)  --  seed 42")
A("=" * 88)
A(f"Primary model: {PRIMARY}   |   Test samples: {N}")
A("")
A("Setup: some promoters appear in the test set under >= 2 different biological")
A("contexts (variety / tissue / stage). The promoter sequence is identical across")
A("those rows; only the context differs. We check whether the model predicts a")
A("different regulation direction (up vs. down) for the same promoter across its")
A("real contexts.")
A("")
A("-" * 88)
A("KEY RESULTS")
A("-" * 88)
A(f"  Unique promoter sequences in test set:              {n_unique}")
A(f"  Promoters appearing under >= 2 real contexts:       {n_multi}")
A(f"  ... with a change in ACTUAL (ground-truth) label:   {n_label_flip} "
  f"({100*n_label_flip/n_multi:.1f}%)")
A(f"  ... with a change in {PRIMARY} PREDICTED direction: "
  f"{pg['n_pred_direction_change']} ({100*pg['frac_pred_direction_change']:.1f}%)")
A("")
A("-" * 88)
A("PER-MODEL PREDICTED DIRECTION-CHANGE RATE (same natural setup)")
A("-" * 88)
A(f"  {'Model':<24}{'# changed':>12}{'% changed':>12}")
for name in MODEL_ORDER:
    r = per_model[name]
    A(f"  {name:<24}{r['n_pred_direction_change']:>12}{100*r['frac_pred_direction_change']:>11.1f}%")
A("")
A("Interpretation: for the same promoter sequence, changing only the biological")
A("context changes the predicted regulation direction for a substantial fraction of")
A("multi-context promoters. This shows the model's prediction depends on biological")
A("context, not the promoter sequence alone -- validating the context-conditional")
A("design. The actual-label change rate is the real-data reference.")
A("")
A("-" * 88)
A("DOES THE MODEL FLIP *CORRECTLY*?")
A("-" * 88)
A(f"  Of the {n_multi} multi-context promoters:")
A(f"    - {n_var} have a DIFFERENT actual label across contexts (context truly matters)")
A(f"    - {n_const} have the SAME actual label across contexts (should NOT flip)")
A("")
A(f"  On the {n_var} variable-label promoters -- fraction the model gets the")
A(f"  direction right in EVERY context (a correct flip); and fraction where it")
A(f"  ignores context and never flips (missed):")
A(f"  On the {n_const} constant-label promoters -- fraction the model wrongly flips")
A(f"  (spurious); lower is better.")
A("")
A(f"  {'Model':<22}{'CorrectFlip':>12}{'Missed':>9}{'RowAcc(var)':>12}{'Spurious':>10}{'RowAcc(con)':>12}")
for name in MODEL_ORDER:
    r = per_model[name]
    A(f"  {name:<22}"
      f"{100*r['var_correct_all_rate']:>11.1f}%"
      f"{100*r['var_missed_rate']:>8.1f}%"
      f"{100*r['var_row_accuracy']:>11.1f}%"
      f"{100*r['const_spurious_flip_rate']:>9.1f}%"
      f"{100*r['const_row_accuracy']:>11.1f}%")
A("")
A("  CorrectFlip = predicts the right direction in every context of a promoter")
A("                whose true label changes (higher is better).")
A("  Missed      = predicts one direction for all contexts, ignoring the true")
A("                context-driven change (lower is better).")
A("  RowAcc(var) = per-sample accuracy on the variable-label multi-context rows.")
A("  Spurious    = flips even though the true label does not change (lower is better).")
A("  RowAcc(con) = per-sample accuracy on the constant-label multi-context rows.")
A("=" * 88)
report = "\n".join(L)
with open(os.path.join(OUT_DIR, "context_flip_report.txt"), "w") as f:
    f.write(report + "\n")
print("\n" + report)
print("\nSaved outputs to:", OUT_DIR)
