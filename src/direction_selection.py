"""Oracle joint-distribution selection; no inferred attribute alignment implied."""
import json
from pathlib import Path

import numpy as np


def select_joint(counts, count, target=None, max_swaps=100):
    """Greedy construction followed by strictly improving one-for-one swaps.

    Each client has equal aggregation weight. Columns must have the SAME
    attribute meaning across clients. Ties use the caller's NumPy RNG scope.
    This is a local search, not a globally optimal subset solver.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or counts.shape[0] == 0:
        raise ValueError("counts must have shape (clients, labels, attributes)")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    totals = counts.sum(axis=(1, 2), keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Each client must contain at least one sample")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= len(counts):
        raise ValueError("count must be an integer in [1, number of clients]")
    if isinstance(max_swaps, (bool, np.bool_)) or not isinstance(max_swaps, (int, np.integer)) or max_swaps < 0:
        raise ValueError("max_swaps must be a nonnegative integer")
    shape = counts.shape[1:]
    target = np.full(shape, 1 / np.prod(shape)) if target is None else np.asarray(target, dtype=float)
    if target.shape != shape or not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1):
        raise ValueError("target must match joint shape and be a probability distribution")
    probabilities = counts / totals

    def loss(total, size):
        return float(np.square(total / size - target).sum())

    def choose(scores):
        scores = np.asarray(scores)
        ties = np.flatnonzero(np.abs(scores - scores.min()) <= 1e-12)
        return int(np.random.choice(ties))

    selected = []
    total = np.zeros(shape)
    for size in range(1, count + 1):
        candidates = [i for i in range(len(counts)) if i not in selected]
        chosen = candidates[choose([loss(total + probabilities[i], size) for i in candidates])]
        selected.append(chosen)
        total += probabilities[chosen]
    greedy_loss = loss(total, count)
    swaps = 0
    for _ in range(max_swaps):
        candidates = [i for i in range(len(counts)) if i not in selected]
        moves = [(position, incoming) for position in range(count) for incoming in candidates]
        if not moves:
            break
        scores = [loss(total - probabilities[selected[pos]] + probabilities[inc], count)
                  for pos, inc in moves]
        best = choose(scores)
        if scores[best] >= loss(total, count) - 1e-12:
            break
        pos, incoming = moves[best]
        total += probabilities[incoming] - probabilities[selected[pos]]
        selected[pos] = incoming
        swaps += 1
    return selected, dict(source="oracle_joint", target=target.tolist(),
                          selected_joint=(total / count).tolist(),
                          greedy_objective=greedy_loss, objective=loss(total, count),
                          swaps=swaps, max_swaps=int(max_swaps))


def record_joint_selection(conf, server_round, available_cids, positions, diagnostics):
    """Keep the objective and actual IDs next to the existing selection trace."""
    directory = Path("checkpoints") / conf["exp_id"]
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(round=server_round, available_cids=available_cids,
               selected_positions=list(map(int, positions)),
               selected_ids=[available_cids[i] for i in positions], **diagnostics)
    with (directory / "joint_selection_trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
