from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "checkpoints" / "local_backup"

DATASETS = {
    "CMNIST GSC": "cmnist_gsc_official_rng_v1",
    "Spawrious GSC": "spawrious_gsc_official_rng_v1",
    "Spawrious GCI": "spawrious_gci_official_rng_v1",
    "Spawrious GAI": "spawrious_gai_official_rng_v1",
}


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float).ravel()
    return entropy(counts / counts.sum()) / math.log(len(counts))


def imbalance(values: np.ndarray) -> float:
    p = np.asarray(values, dtype=float)
    p /= p.sum()
    return 1.0 - entropy(p) / math.log(len(p))


def sc(matrix: np.ndarray) -> float:
    p = matrix.astype(float) / matrix.sum()
    py = p.sum(axis=1)
    pg = p.sum(axis=0)
    mi = 0.0
    for y, g in itertools.product(range(2), repeat=2):
        if p[y, g] > 0:
            mi += p[y, g] * math.log(p[y, g] / (py[y] * pg[g]))
    denom = entropy(py) + entropy(pg)
    return 2.0 * mi / denom if denom else 0.0


def ci(matrix: np.ndarray) -> float:
    return imbalance(matrix.sum(axis=1))


def ai(matrix: np.ndarray) -> float:
    return imbalance(matrix.sum(axis=0))


def signed_phi(matrix: np.ndarray) -> float:
    a, b, c, d = matrix.ravel().astype(float)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denom if denom else 0.0


def structural_type(matrix: np.ndarray) -> str:
    scores = {"SC": sc(matrix), "AI": ai(matrix), "CI": ci(matrix)}
    return max(scores, key=scores.get)


def pairwise_tv(matrices: np.ndarray) -> float:
    distributions = matrices / matrices.sum(axis=(1, 2), keepdims=True)
    vals = [0.5 * np.abs(a - b).sum() for a, b in itertools.combinations(distributions, 2)]
    return float(np.mean(vals)) if vals else 0.0


def complementarity_gain(matrices: np.ndarray) -> float:
    pooled = matrices.sum(axis=0)
    individual = [normalized_entropy(m) for m in matrices]
    return normalized_entropy(pooled) - float(np.mean(individual))


def fsc(matrices: np.ndarray) -> float:
    # Matches src/corr.py: average conditional-distribution disagreement.
    cond = matrices / matrices.sum(axis=2, keepdims=True)
    total = sum(np.abs(a - b).sum() for a, b in itertools.combinations(cond, 2))
    k, y = len(matrices), matrices.shape[1]
    return float(total / (k * (k - 1) * y))


def dsi(matrices: np.ndarray) -> float:
    sizes = matrices.sum(axis=(1, 2)).astype(float)
    proportions = sizes / sizes.sum()
    return float(1.0 - len(sizes) * np.exp(np.log(proportions).mean()))


def read_config(path: Path) -> tuple[int, str]:
    text = path.read_text(encoding="utf-8")
    seed = int(re.search(r"(?m)^seed:\s*(\d+)", text).group(1))
    method = re.search(r"(?m)^\s+selection_method:\s*(\S+)", text).group(1)
    return seed, "FedDiverse" if method == "triplets_stochasticmatrix" else "Random"


def read_client_info(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = sorted(map(int, data))
    true_matrices, predicted = [], []
    for cid in ids:
        row = data[str(cid)]
        true_matrices.append(np.array([row[f"interaction_matrix_{i}"] for i in range(4)]).reshape(2, 2))
        predicted.append([row.get("SC", np.nan), row.get("AI", np.nan), row.get("CI", np.nan)])
    return np.array(true_matrices), np.array(predicted)


def read_selection(path: Path, method: str) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", dtype=int)
    if method == "FedDiverse":
        assert np.all(rows[:2].sum(axis=1) == rows.shape[1])
        rows = rows[2:]
    assert np.all(rows.sum(axis=1) == 9)
    return rows


def parse_summaries(dataset_root: Path) -> dict[tuple[str, int], tuple[float, float]]:
    result = {}
    for path in dataset_root.glob("logs/**/*summary.txt"):
        text = path.read_text(encoding="utf-8")
        method_match = re.search(r"(?m)^method=(.+)$", text)
        if not method_match:
            continue
        method = method_match.group(1).strip()
        for seed, acc, worst in re.findall(
            r"seed=(\d+), test_accuracy=([\d.]+), worst_group=([\d.]+)", text
        ):
            result[(method, int(seed))] = (float(acc), float(worst))
    return result


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def selection_metrics(rows: np.ndarray, matrices: np.ndarray) -> dict[str, float]:
    types = np.array([structural_type(m) for m in matrices])
    round_records = []
    for row in rows:
        ids = np.flatnonzero(row)
        selected = matrices[ids]
        pooled = selected.sum(axis=0)
        counts = Counter(types[ids])
        round_records.append(
            {
                "group_entropy": normalized_entropy(pooled),
                "min_group_exposure": float(pooled.min() / (pooled.sum() / 4.0)),
                "pairwise_tv": pairwise_tv(selected),
                "complementarity_gain": complementarity_gain(selected),
                "type_coverage": len(counts),
                "selected_GSC": sc(pooled),
                "selected_GCI": ci(pooled),
                "selected_GAI": ai(pooled),
            }
        )
    out = {k: float(np.mean([r[k] for r in round_records])) for k in round_records[0]}
    freq = rows.sum(axis=0).astype(float)
    p = freq / freq.sum()
    out.update(
        {
            "selection_entropy": entropy(p) / math.log(len(p)),
            "effective_clients": math.exp(entropy(p)),
            "frequency_cv": float(freq.std() / freq.mean()),
            "frequency_min": int(freq.min()),
            "frequency_max": int(freq.max()),
        }
    )
    for kind in ("SC", "AI", "CI"):
        out[f"share_{kind}"] = float(rows[:, types == kind].sum() / rows.sum())
    return out


def fmt(v: float, digits: int = 3) -> str:
    if np.isnan(v):
        return "NA"
    return f"{v:.{digits}f}"


dataset_rows = []
run_rows = []
client_rows = []
round_selection_rows = []

for dataset, folder in DATASETS.items():
    dataset_root = BACKUP / folder
    run_dirs = sorted((dataset_root / "checkpoints").iterdir())
    parsed_results = parse_summaries(dataset_root)
    canonical_matrices = None

    for run_dir in run_dirs:
        seed, method = read_config(run_dir / "config.yaml")
        matrices, predicted = read_client_info(run_dir / "client_info.json")
        rows = read_selection(run_dir / "client_weights.csv", method)
        metrics = selection_metrics(rows, matrices)
        types = np.array([structural_type(m) for m in matrices])
        for update_round, selection_row in enumerate(rows, start=1):
            ids = np.flatnonzero(selection_row)
            selected = matrices[ids]
            pooled_selected = selected.sum(axis=0)
            type_counts = Counter(types[ids])
            round_selection_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "update_round": update_round,
                    "selected_client_ids": " ".join(map(str, ids)),
                    "SC_clients": type_counts["SC"],
                    "AI_clients": type_counts["AI"],
                    "CI_clients": type_counts["CI"],
                    "group_entropy": normalized_entropy(pooled_selected),
                    "min_group_exposure": float(pooled_selected.min() / (pooled_selected.sum() / 4.0)),
                    "pairwise_tv": pairwise_tv(selected),
                    "complementarity_gain": complementarity_gain(selected),
                }
            )
        true_triplets = np.array([[sc(m), ai(m), ci(m)] for m in matrices])
        correlations = [pearson(true_triplets[:, i], predicted[:, i]) for i in range(3)]
        acc, worst = parsed_results[(method, seed)]
        record = {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "rounds_analyzed": len(rows),
            **metrics,
            "dht_sc_r": correlations[0],
            "dht_ai_r": correlations[1],
            "dht_ci_r": correlations[2],
            "global_accuracy": acc,
            "worst_group_accuracy": worst,
        }
        run_rows.append(record)

        freq = rows.sum(axis=0)
        types = [structural_type(m) for m in matrices]
        for cid, (matrix, pred, count, kind) in enumerate(zip(matrices, predicted, freq, types)):
            client_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "client_id": cid,
                    "datasize": int(matrix.sum()),
                    "n_y0g0": int(matrix[0, 0]),
                    "n_y0g1": int(matrix[0, 1]),
                    "n_y1g0": int(matrix[1, 0]),
                    "n_y1g1": int(matrix[1, 1]),
                    "true_type": kind,
                    "true_SC": sc(matrix),
                    "true_AI": ai(matrix),
                    "true_CI": ci(matrix),
                    "signed_phi": signed_phi(matrix),
                    "predicted_SC": pred[0],
                    "predicted_AI": pred[1],
                    "predicted_CI": pred[2],
                    "selected_rounds": int(count),
                    "selection_rate": float(count / len(rows)),
                }
            )
        if method == "FedDiverse" and seed == 42:
            canonical_matrices = matrices

    assert canonical_matrices is not None
    pooled = canonical_matrices.sum(axis=0)
    types = Counter(structural_type(m) for m in canonical_matrices)
    dataset_rows.append(
        {
            "dataset": dataset,
            "clients": len(canonical_matrices),
            "samples": int(pooled.sum()),
            "y0g0": int(pooled[0, 0]),
            "y0g1": int(pooled[0, 1]),
            "y1g0": int(pooled[1, 0]),
            "y1g1": int(pooled[1, 1]),
            "GSC": sc(pooled),
            "LSC": float(np.mean([sc(m) for m in canonical_matrices])),
            "FSC": fsc(canonical_matrices),
            "GCI": ci(pooled),
            "LCI": float(np.mean([ci(m) for m in canonical_matrices])),
            "GAI": ai(pooled),
            "LAI": float(np.mean([ai(m) for m in canonical_matrices])),
            "DSI": dsi(canonical_matrices),
            "pairwise_tv": pairwise_tv(canonical_matrices),
            "complementarity_gain": complementarity_gain(canonical_matrices),
            "SC_clients": types["SC"],
            "AI_clients": types["AI"],
            "CI_clients": types["CI"],
        }
    )


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


write_csv(ROOT / "results" / "four_datasets_dataset_metrics.csv", dataset_rows)
write_csv(ROOT / "results" / "four_datasets_round_selection_metrics.csv", run_rows)
write_csv(ROOT / "results" / "four_datasets_client_selection.csv", client_rows)
write_csv(ROOT / "results" / "four_datasets_exact_round_selections.csv", round_selection_rows)


# Add paired gains to the run-level records for report generation.
for row in run_rows:
    if row["method"] == "FedDiverse":
        random_row = next(
            r for r in run_rows
            if r["dataset"] == row["dataset"] and r["seed"] == row["seed"] and r["method"] == "Random"
        )
        row["accuracy_gain"] = row["global_accuracy"] - random_row["global_accuracy"]
        row["worst_group_gain"] = row["worst_group_accuracy"] - random_row["worst_group_accuracy"]


md = []
md.append("# 四个已完成数据集的异质性—选择—性能分析\n")
md.append(
    "> 口径：只分析已冻结的 seed 42/43/44 最终 checkpoint，不补跑、不挑中间 checkpoint。"
    "FedDiverse 的逐轮统计排除全客户端预训练轮与 DHT-only 轮，只保留 199 个选择更新轮；"
    "Random 使用全部 200 个更新轮。所有准确率单位均为百分点。\n"
)
md.append("## 1. 数据异质性与潜在互补性\n")
md.append("| Dataset | pooled [y0g0,y0g1,y1g0,y1g1] | GSC / GCI / GAI | local LSC / LCI / LAI | FSC | DSI | SC/AI/CI clients | pairwise TV | complementarity gain |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for r in dataset_rows:
    md.append(
        f"| {r['dataset']} | [{r['y0g0']},{r['y0g1']},{r['y1g0']},{r['y1g1']}] | "
        f"{fmt(r['GSC'])} / {fmt(r['GCI'])} / {fmt(r['GAI'])} | "
        f"{fmt(r['LSC'])} / {fmt(r['LCI'])} / {fmt(r['LAI'])} | {fmt(r['FSC'])} | {fmt(r['DSI'])} | "
        f"{r['SC_clients']}/{r['AI_clients']}/{r['CI_clients']} | {fmt(r['pairwise_tv'])} | {fmt(r['complementarity_gain'])} |"
    )

md.append("\n指标解释：G/L 分别表示 pooled-global 与客户端均值；FSC 衡量客户端条件分布差异；DSI 衡量客户端大小不等；pairwise TV 越大表示客户端联合分布差异越明显；complementarity gain 是合并后的四群体归一化熵减去客户端平均熵，越大表示客户端合并确实互补。\n")

md.append("## 2. FedDiverse 实际选择与 Random 对照\n")
md.append("以下均为三个 seed 的均值；括号内是 FedDiverse 相对 Random 的变化。\n")
md.append("| Dataset | type share SC/AI/CI | type coverage | pairwise TV | complementarity gain | group entropy | min-group exposure | freq CV | effective clients |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for dataset in DATASETS:
    frows = [r for r in run_rows if r["dataset"] == dataset and r["method"] == "FedDiverse"]
    rrows = [r for r in run_rows if r["dataset"] == dataset and r["method"] == "Random"]
    mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
    cell = lambda key: f"{fmt(mean(frows,key))} ({mean(frows,key)-mean(rrows,key):+.3f})"
    md.append(
        f"| {dataset} | {mean(frows,'share_SC'):.1%}/{mean(frows,'share_AI'):.1%}/{mean(frows,'share_CI'):.1%} | "
        f"{cell('type_coverage')} | {cell('pairwise_tv')} | {cell('complementarity_gain')} | "
        f"{cell('group_entropy')} | {cell('min_group_exposure')} | {cell('frequency_cv')} | {cell('effective_clients')} |"
    )

md.append("\nmin-group exposure = 每轮所选样本中最小群体计数 / 四群体完全均衡时的计数；1 表示完全均衡。frequency CV 越小、effective clients 越接近总客户端数，表示长期选择越均匀。\n")

md.append("### FedDiverse 高频与低频客户端\n")
md.append("| Dataset | seed | top-5 client(type):count | never selected | min/max count |")
md.append("|---|---:|---|---|---:|")
for dataset in DATASETS:
    for seed in (42, 43, 44):
        rows = [r for r in client_rows if r["dataset"] == dataset and r["seed"] == seed and r["method"] == "FedDiverse"]
        ranked = sorted(rows, key=lambda r: (-r["selected_rounds"], r["client_id"]))
        top = ", ".join(f"{r['client_id']}({r['true_type']}):{r['selected_rounds']}" for r in ranked[:5])
        never = [str(r["client_id"]) for r in rows if r["selected_rounds"] == 0]
        counts = [r["selected_rounds"] for r in rows]
        md.append(f"| {dataset} | {seed} | {top} | {', '.join(never) if never else 'none'} | {min(counts)}/{max(counts)} |")
md.append("\n每一轮的完整客户端 ID、类型构成和 diversity 指标见 `four_datasets_exact_round_selections.csv`。\n")

md.append("## 3. 每个 seed 的选择质量与最终性能\n")
md.append("| Dataset | FedDiverse global acc | Δ vs Random | FedDiverse worst-group | Δ vs Random |")
md.append("|---|---:|---:|---:|---:|")
for dataset in DATASETS:
    frows = [r for r in run_rows if r["dataset"] == dataset and r["method"] == "FedDiverse"]
    rrows = [r for r in run_rows if r["dataset"] == dataset and r["method"] == "Random"]
    f_acc = np.mean([r["global_accuracy"] for r in frows])
    r_acc = np.mean([r["global_accuracy"] for r in rrows])
    f_worst = np.mean([r["worst_group_accuracy"] for r in frows])
    r_worst = np.mean([r["worst_group_accuracy"] for r in rrows])
    md.append(f"| {dataset} | {f_acc:.3f} | {f_acc-r_acc:+.3f} | {f_worst:.3f} | {f_worst-r_worst:+.3f} |")
md.append("")
md.append("| Dataset | seed | Δ group entropy | Δ min exposure | Δ pairwise TV | DHT r(SC/AI/CI) | global acc F/R (gain) | worst-group F/R (gain) |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for frow in [r for r in run_rows if r["method"] == "FedDiverse"]:
    rrow = next(r for r in run_rows if r["dataset"] == frow["dataset"] and r["seed"] == frow["seed"] and r["method"] == "Random")
    md.append(
        f"| {frow['dataset']} | {frow['seed']} | {frow['group_entropy']-rrow['group_entropy']:+.3f} | "
        f"{frow['min_group_exposure']-rrow['min_group_exposure']:+.3f} | {frow['pairwise_tv']-rrow['pairwise_tv']:+.3f} | "
        f"{fmt(frow['dht_sc_r'],2)}/{fmt(frow['dht_ai_r'],2)}/{fmt(frow['dht_ci_r'],2)} | "
        f"{frow['global_accuracy']:.3f}/{rrow['global_accuracy']:.3f} ({frow['accuracy_gain']:+.3f}) | "
        f"{frow['worst_group_accuracy']:.3f}/{rrow['worst_group_accuracy']:.3f} ({frow['worst_group_gain']:+.3f}) |"
    )

md.append("\n## 4. 结论链\n")
md.append("总体上，四组数据都存在真实客户端互补性；FedDiverse 也都提高了单轮 pairwise TV，但同时把长期选择集中到更少的有效客户端。决定成败的不是‘有没有 diversity’，而是 diversity 是否沿着当前数据集的主异质性方向、并持续覆盖真正困难的群体。DHT 诊断进一步显示：CI 的估计相关系数始终为 1；AI 基本接近 0 或为负；SC 从 CMNIST 的 0.90–0.95，下降到 Spawrious GSC/GCI 的不稳定中等相关，在 GAI 更变为负相关。这与 CMNIST 最稳定、GAI 最失败的结果一致，但仍只是机制证据而非严格因果证明。\n")
md.append("- **CMNIST GSC**：全局以 SC 为主，但客户端同时包含 SC/AI/CI 三类，且 pooled 分布近乎均衡，说明互补性强。FedDiverse 稳定提高所选集合的群体均衡与互补性，三个 seed 的 worst-group 都提升，构成四组中最完整、最一致的证据链。")
md.append("- **Spawrious GSC**：结构互补性同样存在，FedDiverse 也确实在三个 seed 中都提高了单轮群体覆盖；但提高幅度随 seed 递减，选择构成对随机轨迹敏感，且预测 AI 与真值 AI 的一致性弱。覆盖改善没有稳定转化为最终鲁棒性，worst-group 三 seed 两正一负、均值与 Random 持平。")
md.append("- **Spawrious GCI**：全局 class imbalance 最强，关键互补来自少数反向 CI 与 AI/SC 客户端。FedDiverse 平均改善选择集合的均衡性，seed 43/44 获得明显 worst-group 提升；seed 42 仍下降，说明有互补性不等于每次随机轨迹都能成功利用。")
md.append("- **Spawrious GAI**：同时存在很强的全局 attribute imbalance、轻微客户端大小不等和复杂混合异质性。虽然客户端间差异最大，FedDiverse 也提高了 pairwise diversity，但最小群体暴露只比 Random 增加 0.007–0.012，几乎没有触及关键短板；三个 seed 中两个显著下降，因此当前选择结果与目标鲁棒性错配。")
md.append("\n## 5. 解释边界\n")
md.append("这些数据支持“异质性 → 可利用的客户端互补性 → 选择集合统计 → 最终性能”的关联分析，但不能声称严格因果：data seed 只固定数据划分，模型初始化、DataLoader、DHT、选择器与 Ray worker RNG 未统一固定；DHT 只在 round 2 估计一次，之后不更新；每个数据集只有三个正式运行。特别是 GAI/GCI，不应把单个 seed 的选择指标与最终性能作强因果解释。")

(ROOT / "results" / "FOUR_DATASET_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("Wrote:")
for name in (
    "four_datasets_dataset_metrics.csv",
    "four_datasets_round_selection_metrics.csv",
    "four_datasets_client_selection.csv",
    "four_datasets_exact_round_selections.csv",
    "FOUR_DATASET_ANALYSIS.md",
):
    print(ROOT / "results" / name)
