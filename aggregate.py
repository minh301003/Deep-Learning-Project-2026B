"""Gộp log từ results/**/ -> analysis/: experiment_log.csv, tables/, figures/, <E1..E4>/.

Chạy từ thư mục gốc:  python aggregate.py
Mọi bảng/hình/số trong báo cáo phải lấy từ đầu ra của file này.

Quy tắc "rõ vs trong nhiễu" (chốt trước khi chạy, xem kế hoạch mục 4):
  sigma_pooled = căn bậc hai trung bình phương sai (qua các seed) của các cấu hình trong cùng nhóm
  (nhóm = một thí nghiệm; E4: theo từng batch size). sigma_delta = sqrt((sigma_a^2 + sigma_b^2) / 2).
  "rõ" khi |Δ mean| >= 2 sigma_delta VÀ Δ cùng dấu ở mọi seed chung; ngược lại "trong nhiễu".
  Đây là quy tắc sàng lọc, không phải kiểm định thống kê chính thức.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

RESULTS, OUT = "results", "analysis"
TABLES, FIGS = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")

E1 = ["E1-sgd", "E1-sgdm", "E1-nesterov", "E1-rmsprop", "E1-adam", "E1-adamw"]
E2 = ["E2-const", "E2-step", "E2-cos", "E2-const-wu", "E2-step-wu", "E2-cos-wu"]
E3 = ["E3-R0", "E3-R1-wd", "E3-R2-do0.1", "E3-R3-do0.3", "E3-R4-do0.5", "E3-R5-flip", "E3-R6-crop",
      "E3-R7-jitter", "E3-R8-augall", "E3-C-combined"]
NORMS, BATCHES = ["bn", "ln", "gn"], [8, 32, 128]
E4 = [f"E4-{n}-b{b}" for n in NORMS for b in BATCHES]
ORDER = E1 + E2 + E3 + E4
EXPERIMENTS = {"E1": E1, "E2": E2, "E3": E3, "E4": E4}

# Bảng màu phân loại (thứ tự cố định, theo skill dataviz): xanh dương, cam, aqua, vàng, hồng, xanh lá
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2.0,
})



# Đọc dữ liệu
def _read(name: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{RESULTS}/**/{name}", recursive=True)):
        d = pd.read_csv(f)
        d["source"] = os.path.relpath(os.path.dirname(f), RESULTS)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load():
    """Đọc và nối runs/epochs/steps/d1 từ mọi thư mục con của results/."""
    runs = _read("runs.csv")
    if runs.empty:
        sys.exit(f"Không tìm thấy {RESULTS}/**/runs.csv")
    runs["exp"] = runs["config"].str.split("-").str[0]
    runs = runs.sort_values("start_time", kind="stable").reset_index(drop=True)
    epochs, steps, d1 = _read("epochs.csv"), _read("steps.csv"), _read("d1.csv")
    if not epochs.empty:    # lượt chạy lại sau khi crash có thể ghi trùng epoch: giữ bản cuối
        epochs = epochs.drop_duplicates(["config", "seed", "epoch"], keep="last")
    if not steps.empty:
        steps = steps.drop_duplicates(["config", "seed", "step"], keep="last")
    return runs, epochs, steps, d1


def valid(runs: pd.DataFrame) -> pd.DataFrame:
    """Lượt completed (bỏ mọi config có hậu tố -rerun), mỗi (config, seed) một dòng (bản mới nhất)."""
    v = runs[(runs.status == "completed") & ~runs.config.str.endswith("-rerun")]
    return v.drop_duplicates(["config", "seed"], keep="last")



# Thống kê và quy tắc nhiễu
def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Theo cấu hình: n, mean, std (ddof=1), vals {seed: giá trị}."""
    rows = []
    for cfg, g in df.groupby("config"):
        v = g.set_index("seed")[metric].dropna()
        rows.append(dict(config=cfg, n=len(v), mean=v.mean() if len(v) else np.nan,
                         std=v.std(ddof=1) if len(v) > 1 else np.nan,
                         vals={int(k): float(x) for k, x in v.items()}))
    out = pd.DataFrame(rows, columns=["config", "n", "mean", "std", "vals"]).set_index("config")
    return out.loc[sorted(out.index, key=_rank)]


def _rank(cfg: str) -> int:
    return ORDER.index(cfg) if cfg in ORDER else len(ORDER)


def group_of(config: str) -> str:
    """Nhóm tính sigma_pooled: một thí nghiệm; E4 theo từng batch size."""
    return "E4-" + config.split("-")[-1] if config.startswith("E4-") else config.split("-")[0]


def sigma_pooled(summ: pd.DataFrame) -> dict:
    """sigma_pooled theo nhóm = sqrt(trung bình phương sai) của các cấu hình có đủ 2 seed."""
    var = {}
    for cfg, r in summ.iterrows():
        if r["n"] >= 2:
            var.setdefault(group_of(cfg), []).append(r["std"] ** 2)
    out = {g: float(np.sqrt(np.mean(v))) for g, v in var.items()}
    for cfg in summ.index:
        out.setdefault(group_of(cfg), np.nan)
    return out


def compare_vals(va: dict, vb: dict, sig_a: float, sig_b: float) -> dict:
    """So hai tập giá trị theo seed. Δ = mean(a) − mean(b) (điểm phần trăm)."""
    common = sorted(set(va) & set(vb))
    res = dict(delta=np.nan, sigma_delta=np.nan, delta_per_seed="", conclusion="không đủ seed")
    if len(va) < 2 or len(vb) < 2 or len(common) < 2 or np.isnan(sig_a) or np.isnan(sig_b):
        if va and vb:
            res["delta"] = float(np.mean(list(va.values())) - np.mean(list(vb.values())))
        return res
    per_seed = {s: va[s] - vb[s] for s in common}
    delta = float(np.mean(list(va.values())) - np.mean(list(vb.values())))
    sig = float(np.sqrt((sig_a ** 2 + sig_b ** 2) / 2))
    same_sign = all(d > 0 for d in per_seed.values()) or all(d < 0 for d in per_seed.values())
    res.update(delta=delta, sigma_delta=sig,
               delta_per_seed="; ".join(f"s{s}={d:+.2f}" for s, d in per_seed.items()),
               conclusion="rõ" if (abs(delta) >= 2 * sig and same_sign) else "trong nhiễu")
    return res


def compare(summ: pd.DataFrame, sigma: dict, a: str, b: str) -> dict:
    va = summ.loc[a, "vals"] if a in summ.index else {}
    vb = summ.loc[b, "vals"] if b in summ.index else {}
    r = compare_vals(va, vb, sigma.get(group_of(a), np.nan), sigma.get(group_of(b), np.nan))
    return dict(a=a, b=b, **r)



# Bảng
def _md(df: pd.DataFrame, digits: int = 2) -> str:
    cols = list(df.columns)
    fmt = lambda x: (f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) and not np.isnan(x)
                     else ("" if isinstance(x, (float, np.floating)) else str(x)))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str) -> None:
    """Ghi .csv (số đầy đủ) và .md (làm tròn)."""
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    with open(os.path.join(TABLES, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(_md(df))
    print(f"  bảng {name}")


def _fmt_pm(m, s) -> str:
    return "" if np.isnan(m) else (f"{m:.2f} ± {s:.2f}" if not np.isnan(s) else f"{m:.2f}")


def summary_table(runs, v, exp_cfgs, extra=()) -> pd.DataFrame:
    """Bảng tóm tắt theo cấu hình: test acc mean ± std, từng seed, val acc, train loss, thời gian, ..."""
    summ = summarize(v[v.config.isin(exp_cfgs)], "final_test_acc")
    rows = []
    for cfg in [c for c in exp_cfgs if c in summ.index or (runs.config == c).any()]:
        g = v[v.config == cfg]
        r = summ.loc[cfg] if cfg in summ.index else None
        row = dict(config=cfg, n_seeds=int(r["n"]) if r is not None else 0,
                   n_hỏng=int(((runs.config == cfg) & (runs.status != "completed")).sum()),
                   test_acc=_fmt_pm(r["mean"], r["std"]) if r is not None else "",
                   test_acc_mean=r["mean"] if r is not None else np.nan,
                   test_acc_std=r["std"] if r is not None else np.nan,
                   từng_seed="; ".join(f"s{s}={x:.2f}" for s, x in r["vals"].items()) if r is not None else "",
                   val_acc_mean=g.final_val_acc.mean(), train_loss_mean=g.final_train_loss.mean(),
                   phút_mean=g.duration_min.mean())
        for c in extra:
            row[c + "_mean"] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare_table(summ, sigma, pairs) -> pd.DataFrame:
    return pd.DataFrame([compare(summ, sigma, a, b) for a, b in pairs]).rename(
        columns={"a": "cấu_hình_a", "b": "cấu_hình_b", "delta": "Δ_điểm_%", "sigma_delta": "σ_Δ",
                 "delta_per_seed": "Δ_từng_seed", "conclusion": "kết_luận"})


# ----------------------------------------------------------------------------------------------
# Hình
# ----------------------------------------------------------------------------------------------
def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f"{name}.png"))
    plt.close(fig)
    print(f"  hình {name}")


def _bars(ax, v, cfgs, labels, ylabel):
    """Cột mean + chấm từng seed."""
    summ = summarize(v[v.config.isin(cfgs)], "final_test_acc")
    for i, c in enumerate([c for c in cfgs if c in summ.index]):
        r = summ.loc[c]
        ax.bar(i, r["mean"], width=0.6, color=COLORS[i % 6], alpha=0.85)
        ax.scatter([i + (j - 0.5) * 0.12 for j in range(len(r["vals"]))], list(r["vals"].values()),
                   color=INK, s=14, zorder=3)
    present = [c for c in cfgs if c in summ.index]
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([labels[c] for c in present], rotation=20)
    lo = min(summ["mean"].min(), min(min(x) for x in summ["vals"].map(lambda d: list(d.values()))))
    ax.set_ylim(max(0, lo - 5), 100)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig_e1(v, epochs):
    d = epochs[epochs.config.isin(E1)]
    if not d.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for i, c in enumerate(E1):
            g = d[d.config == c].groupby("epoch")[["val_acc", "train_loss"]].mean()
            if g.empty:
                continue
            axes[0].plot(g.index, g.val_acc, color=COLORS[i], label=c[3:])
            axes[1].plot(g.index, g.train_loss, color=COLORS[i], label=c[3:])
        axes[0].set(xlabel="Epoch", ylabel="Độ chính xác val (%)", title="Val acc theo epoch (TB các seed)")
        axes[1].set(xlabel="Epoch", ylabel="Train loss", yscale="log", title="Train loss theo epoch (TB các seed)")
        axes[0].set_ylim(40, 100)
        axes[0].legend(ncol=2)
        _save(fig, "F1_e1_epoch_curves")
    if not v[v.config.isin(E1)].empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        _bars(ax, v, E1, {c: c[3:] for c in E1}, "Test acc cuối (%)")
        ax.set_title("E1: test acc epoch 20 (cột = TB, chấm = từng seed)")
        _save(fig, "F2_e1_final_acc")


def fig_e2(v, steps):
    d = steps[steps.config.isin(E2)]
    if not d.empty:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw=dict(height_ratios=[3, 1.3, 2.2]))
        for i, c in enumerate(E2):
            g = d[d.config == c].groupby("step")[["loss", "lr"]].mean()
            if g.empty:
                continue
            col = COLORS[i % 3]
            ls = "--" if c.endswith("-wu") else "-"
            sm = g.loss.ewm(span=100).mean()
            axes[0].plot(g.index, sm, color=col, ls=ls, lw=1.6, label=c[3:])
            axes[1].plot(g.index, g.lr, color=col, ls=ls, lw=1.6)
            z = g.iloc[:1000]
            axes[2].plot(z.index, z.loss.ewm(span=20).mean(), color=col, ls=ls, lw=1.6)
        for ax in axes[:2]:
            for x in (3510, 5265):
                ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        axes[0].set(ylabel="Train loss (EMA)", yscale="log",
                    title="E2: loss theo bước (TB các seed; nét đứt = có warm-up; vạch chấm = mốc step decay)")
        axes[1].set(ylabel="Learning rate")
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[2].set(xlabel="Bước", ylabel="Train loss (EMA)", title="Phóng to 1 000 bước đầu (warm-up)")
        axes[0].legend(ncol=3)
        _save(fig, "F3_e2_loss_per_step")
    if not v[v.config.isin(E2)].empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        _bars(ax, v, E2, {c: c[3:] for c in E2}, "Test acc cuối (%)")
        ax.set_title("E2: test acc epoch 20 của 6 lịch LR")
        _save(fig, "F4_e2_final_acc")


def fig_e3(v, epochs, comp, sigma, runs):
    if not comp.empty and comp["Δ_điểm_%"].notna().any():          # F5
        c = comp.dropna(subset=["Δ_điểm_%"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        y = np.arange(len(c))
        sd = c["σ_Δ"].dropna().iloc[0] if c["σ_Δ"].notna().any() else 0
        ax.axvspan(-2 * sd, 2 * sd, color="#efeee9", label="±2σ_Δ (vùng nhiễu)")
        ax.barh(y, c["Δ_điểm_%"], color=[COLORS[0] if k == "rõ" else "#8a8984" for k in c["kết_luận"]], height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([s[3:] for s in c["cấu_hình_a"]])
        ax.invert_yaxis()
        ax.axvline(0, color=INK, lw=0.8)
        ax.set(xlabel="Δ test acc so với R0 (điểm %)", title="E3: tác động từng kỹ thuật so với không chính quy hóa (xanh = rõ)")
        ax.legend(loc="lower right")
        _save(fig, "F5_e3_delta_vs_R0")
    d = epochs[epochs.config.isin(E3)]
    if not d.empty:                                                  # F6
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, c in enumerate(["E3-R0", "E3-R1-wd", "E3-R8-augall", "E3-C-combined"]):
            g = d[d.config == c].groupby("epoch")[["train_eval_acc", "val_acc"]].mean()
            if g.empty:
                continue
            ax.plot(g.index, g.train_eval_acc, color=COLORS[i], label=f"{c[3:]} train-eval")
            ax.plot(g.index, g.val_acc, color=COLORS[i], ls="--")
        es = runs[(runs.config == "E3-R0") & (runs.status == "completed")].es_stop_epoch.mean()
        if not np.isnan(es):
            ax.axvline(es, color=MUTED, ls=":", lw=1)
            ax.text(es, 50, f" ES của R0 (TB epoch {es:.1f})", color=MUTED)
        ax.set(xlabel="Epoch", ylabel="Độ chính xác (%)", ylim=(40, 101),
               title="E3: train-eval (liền) vs val (đứt): khoảng cách = overfit")
        ax.legend(ncol=2, loc="lower right")
        _save(fig, "F6_e3_overfit_gap")
    dr = [("E3-R0", 0.0), ("E3-R2-do0.1", 0.1), ("E3-R3-do0.3", 0.3), ("E3-R4-do0.5", 0.5)]
    vv = v[v.config.isin([c for c, _ in dr])]
    if not vv.empty:                                                 # F7
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, lab in zip(axes, ["final_test_acc", "gen_gap"], ["Test acc cuối (%)", "gen_gap = train-eval − test (điểm %)"]):
            s = summarize(vv, metric)
            xs = [p for c, p in dr if c in s.index]
            ax.errorbar(xs, [s.loc[c, "mean"] for c, _ in dr if c in s.index],
                        yerr=[0 if np.isnan(s.loc[c, "std"]) else s.loc[c, "std"] for c, _ in dr if c in s.index],
                        color=COLORS[0], marker="o", capsize=3)
            ax.set(xlabel="Tỉ lệ dropout p", ylabel=lab, xticks=[p for _, p in dr])
        axes[0].set_title("Test acc theo dropout (không aug, không WD)")
        axes[1].set_title("gen_gap theo dropout")
        _save(fig, "F7_e3_dropout")


def fig_e4(v, d1):
    s = summarize(v[v.config.isin(E4)], "final_test_acc")
    if not s.empty:                                                  # F8
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, n in enumerate(NORMS):
            cfgs = [(b, f"E4-{n}-b{b}") for b in BATCHES if f"E4-{n}-b{b}" in s.index]
            if cfgs:
                ax.errorbar([b for b, _ in cfgs], [s.loc[c, "mean"] for _, c in cfgs],
                            yerr=[0 if np.isnan(s.loc[c, "std"]) else s.loc[c, "std"] for _, c in cfgs],
                            color=COLORS[i], marker="o", capsize=3, label=n.upper())
        ax.set(xscale="log", xticks=BATCHES, xlabel="Batch size", ylabel="Test acc cuối (%)",
               title="E4: test acc theo batch size (LR theo linear scaling)")
        ax.set_xticklabels(BATCHES)
        ax.minorticks_off()
        ax.legend()
        _save(fig, "F8_e4_acc_vs_batch")
    if not d1.empty:                                                 # F9
        g = d1.groupby(["layer", "batch_size"])["mean_noise"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, (layer, gg) in enumerate(g.groupby("layer")):
            ax.plot(gg.batch_size, gg.mean_noise, marker="o", color=COLORS[i], label=layer)
        b = np.array(sorted(g.batch_size.unique()), dtype=float)
        ref = g[g.batch_size == b.min()].mean_noise.max() * (b / b.min()) ** -0.5
        ax.plot(b, ref, color=MUTED, ls="--", lw=1.2, label="độ dốc −½ (tham chiếu)")
        ax.set(xscale="log", yscale="log", xlabel="Batch size B", ylabel="Nhiễu mean theo kênh (đơn vị σ quần thể)",
               title="D1: nhiễu thống kê batch của BN theo B (log–log)")
        ax.legend()
        _save(fig, "F9_e4_bn_noise")


def d1_slopes(d1) -> pd.DataFrame:
    """Độ dốc log–log của mean_noise và var_noise theo B (kỳ vọng ≈ −1/2)."""
    rows = []
    for layer, g in d1.groupby("layer"):
        g = g.groupby("batch_size")[["mean_noise", "var_noise"]].mean().reset_index()
        row = dict(layer=layer)
        for m in ("mean_noise", "var_noise"):
            row[f"slope_{m}"] = float(np.polyfit(np.log(g.batch_size), np.log(g[m]), 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


# Thư mục riêng cho từng thí nghiệm: analysis/<E1..E4>/{runs.csv, epochs.csv, figures/}
def fig_loss_grid(ep: pd.DataFrame, cfgs: list, exp: str) -> None:
    """Lưới: mỗi ô một config; train_loss (xanh) và val_loss (cam) theo epoch.
    Đường đậm = trung bình các seed, đường mờ = từng seed."""
    cfgs = [c for c in cfgs if (ep.config == c).any()]
    if not cfgs:
        return
    ncols = min(3, len(cfgs))
    nrows = int(np.ceil(len(cfgs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, c in zip(axes.flat, cfgs):
        g = ep[ep.config == c]
        for col, key, lab in ((COLORS[0], "train_loss", "train"), (COLORS[1], "val_loss", "val")):
            for _, gs in g.groupby("seed"):
                ax.plot(gs.epoch, gs[key], color=col, lw=0.8, alpha=0.3)
            m = g.groupby("epoch")[key].mean()
            ax.plot(m.index, m.values, color=col, lw=2, label=lab)
        ax.set_title(c[len(exp) + 1:], fontsize=10)
        ax.set_yscale("log")
    for ax in axes.flat[len(cfgs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss")
    axes[0, 0].legend()
    fig.suptitle(f"{exp}: train loss và val loss theo epoch (đậm = TB seed, mờ = từng seed)")
    fig.tight_layout()
    d = os.path.join(OUT, exp, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "loss_curves.png"))
    plt.close(fig)
    print(f"  hình {exp}/figures/loss_curves")


def per_experiment(runs: pd.DataFrame, epochs: pd.DataFrame) -> None:
    for exp, cfgs in EXPERIMENTS.items():
        r = runs[runs.config.isin(cfgs)]
        if r.empty:
            continue
        d = os.path.join(OUT, exp)
        os.makedirs(d, exist_ok=True)
        r.drop(columns=["exp"]).to_csv(os.path.join(d, "runs.csv"), index=False)
        if epochs.empty:
            continue
        ep = epochs[epochs.config.isin(cfgs)].copy()
        ep["_r"] = ep.config.map(_rank)
        ep = ep.sort_values(["_r", "seed", "epoch"]).drop(columns="_r")
        ep.to_csv(os.path.join(d, "epochs.csv"), index=False)
        print(f"  {exp}: {len(r)} lượt, {len(ep)} dòng epoch")
        fig_loss_grid(ep, cfgs, exp)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    runs, epochs, steps, d1 = load()
    v = valid(runs)
    os.makedirs(OUT, exist_ok=True)
    print(f"Đọc {len(runs)} dòng log từ {runs.source.nunique()} nguồn")
    runs.drop(columns=["exp"]).to_csv(os.path.join(OUT, "experiment_log.csv"), index=False)

    metric = "final_test_acc"
    summ = summarize(v, metric)
    sigma = sigma_pooled(summ)
    print("sigma_pooled:", {k: round(x, 3) for k, x in sigma.items()})
    pd.DataFrame([dict(nhóm=k, sigma_pooled=x) for k, x in sigma.items()]).pipe(save_table, "sigma_pooled")

    def has(cfgs):
        return v.config.isin(cfgs).any()

    if has(E1):                                                       # T1
        save_table(summary_table(runs, v, E1, ("es_stop_epoch", "test_acc_at_es")), "T1_optimizer")
        pairs = [("E1-sgdm", "E1-sgd"), ("E1-nesterov", "E1-sgdm"), ("E1-adamw", "E1-adam"),
                 ("E1-rmsprop", "E1-sgdm"), ("E1-adam", "E1-sgdm"), ("E1-adamw", "E1-sgdm")]
        save_table(compare_table(summ, sigma, pairs), "T1_optimizer_so_sanh")
    if has(E2):                                                       # T2
        save_table(summary_table(runs, v, E2), "T2_lr_schedule")
        pairs = [(f"E2-{s}-wu", f"E2-{s}") for s in ("const", "step", "cos")] + \
                [("E2-step", "E2-const"), ("E2-cos", "E2-const"), ("E2-cos", "E2-step")]
        save_table(compare_table(summ, sigma, pairs), "T2_lr_schedule_so_sanh")
    comp3 = pd.DataFrame()
    if has(E3):                                                       # T3
        t3 = summary_table(runs, v, E3, ("gen_gap", "es_stop_epoch", "test_acc_at_es"))
        save_table(t3, "T3_regularization")
        comp3 = compare_table(summ, sigma, [(c, "E3-R0") for c in E3[1:]])
        save_table(comp3, "T3_regularization_so_sanh")
        _e3_extra(v, summ, comp3)
    if has(E4):                                                       # T4
        _e4_tables(v, summ, sigma)
    t5 = runs.groupby(["exp", "owner", "gpu_name"]).agg(số_lượt=("config", "size"),
                                                         giờ_gpu=("duration_min", lambda x: x.sum() / 60)).reset_index()
    save_table(t5, "T5_gio_gpu")                                      # T5
    if not d1.empty:
        save_table(d1_slopes(d1), "D1_do_doc")

    print("Hình:")
    fig_e1(v, epochs)
    if not steps.empty:
        fig_e2(v, steps)
    elif has(E2):
        fig_e2(v, steps)
    fig_e3(v, epochs, comp3, sigma, runs)
    fig_e4(v, d1)

    print("Theo từng thí nghiệm:")
    per_experiment(runs, epochs)


def _e3_extra(v, summ, comp3):
    """Bảng 'tính cộng' và bảng early stopping của E3."""
    c = {r["cấu_hình_a"]: r["Δ_điểm_%"] for _, r in comp3.iterrows()}
    comb = v[v.config == "E3-C-combined"]
    if not comb.empty and "E3-C-combined" in c:
        p = float(json.loads(comb.config_json.iloc[0])["dropout"])
        do = {0.1: "E3-R2-do0.1", 0.3: "E3-R3-do0.3", 0.5: "E3-R4-do0.5"}.get(round(p, 3))
        parts = {"WD (R1)": c.get("E3-R1-wd"), f"Dropout p*={p} ({do})": c.get(do), "Aug flip+crop+jitter (R8)": c.get("E3-R8-augall")}
        tot = sum(x for x in parts.values() if x is not None and not np.isnan(x))
        rows = [dict(thành_phần=k, Δ_điểm_=x) for k, x in parts.items()]
        rows += [dict(thành_phần="TỔNG Δ riêng lẻ", Δ_điểm_=tot),
                 dict(thành_phần="Δ của E3-C-combined", Δ_điểm_=c["E3-C-combined"])]
        save_table(pd.DataFrame(rows), "T3_tinh_cong")
    rows = []
    for cfg in E3:
        g = v[v.config == cfg].sort_values("seed")
        if g.empty:
            continue
        rows.append(dict(config=cfg, es_stop_epoch="; ".join(f"s{s}={int(e)}" for s, e in zip(g.seed, g.es_stop_epoch)),
                         Δ_ES_từng_seed="; ".join(f"s{s}={a - b:+.2f}" for s, a, b in zip(g.seed, g.test_acc_at_es, g.final_test_acc)),
                         Δ_ES_mean=float((g.test_acc_at_es - g.final_test_acc).mean())))
    save_table(pd.DataFrame(rows), "T3_early_stopping")


def _e4_tables(v, summ, sigma):
    rows = []
    for n in NORMS:
        row = {"norm": n.upper()}
        for b in BATCHES:
            c = f"E4-{n}-b{b}"
            row[f"B={b}"] = _fmt_pm(summ.loc[c, "mean"], summ.loc[c, "std"]) if c in summ.index else ""
        rows.append(row)
    save_table(pd.DataFrame(rows), "T4_norm_luoi_3x3")
    pairs = [(f"E4-{n}-b{b}", f"E4-bn-b{b}") for b in BATCHES for n in ("ln", "gn")]
    cmp_ = [compare(summ, sigma, a, bb) for a, bb in pairs]
    for n in NORMS:                    # mức giảm acc 128 -> 8 của từng loại norm
        a, b = f"E4-{n}-b128", f"E4-{n}-b8"
        if a in summ.index and b in summ.index:
            r = compare_vals(summ.loc[a, "vals"], summ.loc[b, "vals"], sigma.get("E4-b128", np.nan), sigma.get("E4-b8", np.nan))
            cmp_.append(dict(a=f"{n.upper()}: B128 − B8 (mức giảm)", b="", **r))
    save_table(pd.DataFrame(cmp_).rename(columns={"a": "cấu_hình_a", "b": "cấu_hình_b", "delta": "Δ_điểm_%",
                                                  "sigma_delta": "σ_Δ", "delta_per_seed": "Δ_từng_seed",
                                                  "conclusion": "kết_luận"}), "T4_norm_so_sanh")


if __name__ == "__main__":
    main()
