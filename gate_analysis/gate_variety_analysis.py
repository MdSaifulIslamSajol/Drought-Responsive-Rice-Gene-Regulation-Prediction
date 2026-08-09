# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Gate-value analysis by variety  --  tests the Discussion claim
==============================================================
Claim (main.tex, "Why gating helps"):
  "For samples from well-represented varieties (IR64, N22), the context
   embeddings are well-trained and the gate increases their influence. For
   sparsely represented varieties, the gate suppresses context and leans on the
   more stable motif signal."

In HybridGatedFusion the fused representation is
      h = g (x) m + (1 - g) (x) c
so g is the per-dimension weight on the MOTIF representation m, and (1 - g) is
the weight on the CONTEXT representation c. The claim therefore predicts:
  - well-represented varieties -> HIGHER context weight (1 - g)  [lower g]
  - sparse varieties           -> LOWER  context weight (1 - g)  [higher g]
i.e. a POSITIVE correlation between a variety's sample count and its mean
context weight (1 - g).

We extract the actual gate g for every test sample (mean over its 128 dims),
average per variety, and correlate with variety sample count.

Usage:
    CUDA_VISIBLE_DEVICES=0 python gate_variety_analysis.py
"""
import os
import sys
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
import joblib
from scipy.stats import pearsonr, spearmanr

PROJECT_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2"
V3_DIR = os.path.join(PROJECT_DIR, "global motif counting based/output_global/hybrid_models_v3_reproducible_seed42")
MODELS_DIR = os.path.join(V3_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "global motif counting based/output_global_data")
OUT_DIR = os.path.join(PROJECT_DIR, "gate_analysis", "results")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
_SCRATCH = "/tmp/gate_scratch"; os.makedirs(_SCRATCH, exist_ok=True)
_cwd, _stdout = os.getcwd(), sys.stdout
os.chdir(_SCRATCH)
import rice_hybrid_classifier_global_v2 as v2  # noqa: E402
sys.stdout = _stdout
os.chdir(_cwd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
    cfg = json.load(f)
MOTIF, SEQ = cfg["motif_cols"], cfg["seq_stat_cols"]
VAR, TIS, STG = cfg["variety_cols"], cfg["tissue_cols"], cfg["stage_cols"]
HD, DR = cfg["hidden_dim"], cfg["dropout"]
nM, nV, nT, nS, nQ = cfg["num_motifs"], cfg["num_varieties"], cfg["num_tissues"], cfg["num_stages"], cfg["num_seq_stats"]

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Use ALL splits so per-variety gate means are based on as many samples as
# possible (the gate is a deterministic function of the trained weights; it does
# not "peek" at labels). Sample counts are reported per split too.
splits = {s: pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv")) for s in ["train", "dev", "test"]}
full = pd.concat(splits.values(), ignore_index=True)

model = v2.HybridGatedFusion(nM, nV, nT, nS, nQ, HD, DR)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "HybridGatedFusion.pt"), map_location=device))
model.to(device).eval()

# Forward hook to capture the gate output g = sigmoid(W_g[m;c]+b_g) in (0,1)^128.
_captured = {}
def _hook(mod, inp, out):
    _captured["g"] = out.detach().cpu().numpy()
h = model.gate.register_forward_hook(_hook)


@torch.no_grad()
def gate_per_sample(df, batch=8192):
    motif = df[MOTIF].values.astype(np.float32)
    seq = df[SEQ].values.astype(np.float32)
    comb = scaler.transform(np.hstack([motif, seq]))
    ms, ss = comb[:, :nM].astype(np.float32), comb[:, nM:].astype(np.float32)
    v = df[VAR].values.astype(np.float32); t = df[TIS].values.astype(np.float32); s = df[STG].values.astype(np.float32)
    n = len(df)
    g_mean = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        model(torch.from_numpy(ms[sl]).to(device), torch.from_numpy(ss[sl]).to(device),
              torch.from_numpy(v[sl]).to(device), torch.from_numpy(t[sl]).to(device),
              torch.from_numpy(s[sl]).to(device))
        g_mean[sl] = _captured["g"].mean(axis=1)  # mean weight on MOTIF over 128 dims
    return g_mean


g_motif = gate_per_sample(full)          # weight on motif  (g)
g_context = 1.0 - g_motif                 # weight on context (1 - g)
var_idx = full[VAR].values.argmax(1)
var_names = [VAR[i].replace("Variety_", "") for i in var_idx]

df = pd.DataFrame({"variety": var_names,
                   "motif_weight": g_motif,
                   "context_weight": g_context})
grp = df.groupby("variety").agg(
    n_samples=("motif_weight", "size"),
    mean_motif_weight=("motif_weight", "mean"),
    mean_context_weight=("context_weight", "mean"),
    std_context_weight=("context_weight", "std"),
).reset_index().sort_values("n_samples", ascending=False)

# Correlations: context weight vs sample count (claim => positive).
n = grp["n_samples"].values.astype(float)
cw = grp["mean_context_weight"].values
pear_r, pear_p = pearsonr(np.log10(n), cw)
spear_r, spear_p = spearmanr(n, cw)

grp.to_csv(os.path.join(OUT_DIR, "gate_by_variety.csv"), index=False)

# ---- Also emit a ready-to-paste LaTeX table (display variety names) ----
_disp = {"IR64_Cleaned": "IR64", "N22_Cleaned3": "N22"}
tex = []
tex.append(r"\begin{table}[!t]")
tex.append(r"\centering")
tex.append(r"\begin{threeparttable}")
tex.append(r"\caption{Learned Gate Weights by Rice Variety (\texttt{GatedFusion})}")
tex.append(r"\label{tab:gate_by_variety}")
tex.append(r"\renewcommand{\arraystretch}{1.2}")
tex.append(r"\begin{tabular}{lrcc}")
tex.append(r"\hline")
tex.append(r"\textbf{Variety} & \textbf{Samples} & \textbf{Motif wt.\ ($g$)} & \textbf{Context wt.\ ($1{-}g$)} \\")
tex.append(r"\hline")
for _, r in grp.iterrows():
    name = _disp.get(r["variety"], r["variety"])
    tex.append(f"{name} & {int(r['n_samples']):,} & {r['mean_motif_weight']:.3f} & {r['mean_context_weight']:.3f} \\\\")
tex.append(r"\hline")
overall_g = float(df["motif_weight"].mean())
tex.append(f"\\textbf{{All}} & \\textbf{{{len(full):,}}} & \\textbf{{{overall_g:.3f}}} & \\textbf{{{1-overall_g:.3f}}} \\\\")
tex.append(r"\hline")
tex.append(r"\end{tabular}")
tex.append(r"\begin{tablenotes}\small")
tex.append(r"\item In the fusion $\mathbf{h}=\mathbf{g}\odot\mathbf{m}+(1-\mathbf{g})\odot\mathbf{c}$, "
           r"$g$ is the mean weight on the motif representation and $1{-}g$ the weight on biological "
           r"context (averaged over the 128 gate dimensions and all samples of each variety). Varieties "
           r"are sorted by sample count. Context weight shows no significant dependence on sample size "
           r"(Spearman $\rho=%.2f$, $p=%.2f$)." % (spear_r, spear_p))
tex.append(r"\end{tablenotes}")
tex.append(r"\end{threeparttable}")
tex.append(r"\end{table}")
with open(os.path.join(OUT_DIR, "gate_by_variety_table.tex"), "w") as f:
    f.write("\n".join(tex) + "\n")

L = []
A = L.append
A("=" * 84)
A("GATE-VALUE ANALYSIS BY VARIETY  (HybridGatedFusion, seed 42)")
A("=" * 84)
A("Fusion:  h = g (x) m + (1-g) (x) c   ->   g = weight on MOTIF, (1-g) = weight on CONTEXT")
A(f"Samples analysed: {len(full)} (train+dev+test)")
A("")
A(f"  {'Variety':<14}{'N':>8}{'MotifW (g)':>13}{'ContextW (1-g)':>16}")
A("  " + "-" * 50)
for _, r in grp.iterrows():
    A(f"  {r['variety']:<14}{int(r['n_samples']):>8}{r['mean_motif_weight']:>13.4f}{r['mean_context_weight']:>16.4f}")
A("")
A("-" * 84)
A("CLAIM TEST: is context weight (1-g) higher for better-represented varieties?")
A("-" * 84)
A(f"  Pearson r (log10 N vs context weight): r = {pear_r:+.3f}  (p = {pear_p:.3g})")
A(f"  Spearman rho (N vs context weight):    rho = {spear_r:+.3f}  (p = {spear_p:.3g})")
A("")
top2 = grp.nlargest(2, "n_samples")["mean_context_weight"].mean()
bot3 = grp.nsmallest(3, "n_samples")["mean_context_weight"].mean()
A(f"  Mean context weight, 2 largest varieties (IR64, N22): {top2:.4f}")
A(f"  Mean context weight, 3 smallest varieties:            {bot3:.4f}")
A("")
if spear_r > 0 and spear_p < 0.05:
    A("  => SUPPORTED: context weight rises with variety sample size (claim holds).")
elif spear_r > 0:
    A("  => WEAK/DIRECTIONAL support (positive trend but not significant): hedge as a tendency.")
else:
    A("  => NOT SUPPORTED by the gate values: state the mechanism as a hypothesis, not a result.")
A("=" * 84)
report = "\n".join(L)
with open(os.path.join(OUT_DIR, "gate_variety_report.txt"), "w") as f:
    f.write(report + "\n")
print("\n" + report)
print("\nSaved:", OUT_DIR)
