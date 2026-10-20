"""Opt-in controls shared by every baseline-validation arm.

Historical configurations do not enable these controls. Seeds are derived by
phase, round and client, so reconstructed Ray clients do not repeat a constant
training RNG stream at every round.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np


def options(conf):
    return conf.get("research") or {}


def enabled(conf):
    return bool(options(conf).get("enabled", False))


def derived_seed(conf, *parts):
    payload = json.dumps([int(options(conf)["training_seed"]), *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def seed_driver(conf):
    if not enabled(conf):
        return
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    value = derived_seed(conf, "driver")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@contextmanager
def rng_scope(conf, *parts, use_torch=True):
    if not enabled(conf):
        yield
        return
    value = derived_seed(conf, *parts)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch = None
    cuda_states = None
    if use_torch:
        import torch
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cpu_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        random.seed(value)
        np.random.seed(value)
        if torch is not None:
            torch.manual_seed(value)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, cudnn_deterministic


def choose_pivot(data_splits, mode="legacy"):
    if mode == "legacy":
        return min(data_splits, key=lambda key: data_splits[key][2])
    if mode != "min_count_gap":
        raise ValueError(f"Unknown pivot mode: {mode}")
    # Labels are pseudo-attributes (biased classifier correct/error), not true g.
    # Numeric class ID breaks ties deterministically; sampling weights stay intact.
    def score(key):
        errors = data_splits[key][0].tensors[1]
        zero = int((errors == 0).sum())
        one = int((errors == 1).sum())
        if zero + one == 0:
            raise ValueError("Cannot use an empty class as pivot")
        return abs(zero - one), int(key)
    if not data_splits:
        raise ValueError("No classes available for pivot selection")
    return min(data_splits, key=score)


def map_positions(positions, available_cids, mode="legacy"):
    if mode == "legacy":
        return [str(index) for index in positions]
    if mode != "candidate":
        raise ValueError(f"Unknown client ID mapping mode: {mode}")
    return [available_cids[int(index)] for index in positions]


def guarded_selection(original_matrix, count):
    """Legacy triplet geometry with explicit masks and degenerate fallbacks.

    Keeps the legacy row permutation/rotation and signed cross-product rule;
    it does not introduce absolute alignment or paper-style normalization.
    """
    matrix = np.asarray(original_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 3:
        raise ValueError("DHT must have shape (3, K)")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("DHT values must be finite and nonnegative")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("Client count must be an integer")
    if not 0 <= count <= matrix.shape[1]:
        raise ValueError("Requested client count exceeds available clients")
    if count == 0:
        return []
    matrix = matrix[np.random.permutation(3)].copy()
    np.random.shuffle(matrix)
    selected = []

    def available():
        return np.array([i for i in range(matrix.shape[1]) if i not in selected], dtype=int)

    def tied_choice(scores, candidates, minimize):
        values = scores[candidates]
        best = values.min() if minimize else values.max()
        return int(np.random.choice(candidates[np.abs(values - best) <= 1e-6]))

    while len(selected) < count:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        candidates = available()
        probabilities = matrix[0, candidates]
        mass = probabilities.sum()
        first = int(np.random.choice(candidates, p=probabilities / mass)) if mass > 0 else int(np.random.choice(candidates))
        selected.append(first)
        norms = np.linalg.norm(matrix, axis=0)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        vector1 = matrix[:, first].copy()
        if len(selected) == count:
            break
        second = tied_choice(matrix.T @ vector1, available(), True)
        selected.append(second)
        vector2 = matrix[:, second].copy()
        matrix[:, [first, second]] = 0
        if len(selected) == count:
            break
        normal = np.cross(vector1, vector2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            third = tied_choice(matrix.T @ (normal / norm), available(), False)
        else:
            # No unique orthogonal direction; use a uniform remaining client.
            third = int(np.random.choice(available()))
        selected.append(third)
        matrix[:, third] = 0
        matrix = matrix[[2, 0, 1], :]
    return selected


def record_selection(conf, server_round, available_cids, positions, actual_ids):
    if not enabled(conf):
        return
    directory = Path("checkpoints") / conf["exp_id"]
    row = dict(round=server_round, available_cids=available_cids,
               selected_positions=list(map(int, positions)), selected_ids=actual_ids,
               pivot_mode=conf["server_opt"].get("pivot_mode", "legacy"),
               selector_mode=conf["server_opt"].get("selector_mode", "legacy"),
               client_id_mapping=conf["server_opt"].get("client_id_mapping", "legacy"))
    with (directory / "selection_trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def metadata(dataset, cache=None):
    """Read identities and y/g without decoding all images or touching RNG."""
    from torch.utils.data import Subset
    from src.datasets.dataset_utils import SubsetDataset
    if cache is None:
        cache = {}
    if id(dataset) in cache:
        return cache[id(dataset)]
    if isinstance(dataset, (Subset, SubsetDataset)):
        base = metadata(dataset.dataset, cache)
        rows = [base[int(index)] for index in dataset.indices]
    elif dataset.__class__.__name__ == "Spawrious":
        data_root = Path("datasets").resolve()
        rows = [(str(Path(path).resolve().relative_to(data_root)).replace("\\", "/"),
                 int(y), int(g)) for path, y, g in dataset.subset]
    elif dataset.__class__.__name__ == "CMNIST":
        split = "train" if dataset.subset.train else "test"
        rows = [(f"MNIST/{split}/{i}", int(y), int(g))
                for i, (y, g) in enumerate(zip(dataset.subset.targets, dataset.subset.attr))]
    else:
        raise ValueError("Baseline validation currently supports Spawrious and CMNIST only")
    cache[id(dataset)] = rows
    return rows


def validation_indices(rows, fraction, seed):
    groups = {}
    for i, (_, y, g) in enumerate(rows):
        groups.setdefault((y, g), []).append(i)
    rng = np.random.default_rng(int(seed))
    fraction = float(fraction)
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie strictly between zero and one")
    validation = []
    for key in sorted(groups):
        indices = groups[key]
        n = max(1, int(len(indices) * fraction))
        if n >= len(indices):
            raise ValueError("Each group must retain training samples")
        validation.extend(rng.permutation(indices)[:n].tolist())
    held_out = set(validation)
    retained = [i for i in range(len(rows)) if i not in held_out]
    return retained, sorted(validation)


def validation_split(train_dataset, conf):
    from torch.utils.data import Subset
    retained, validation = validation_indices(metadata(train_dataset),
                                              options(conf)["validation_fraction"],
                                              options(conf)["validation_seed"])
    # Use torch Subset here: the repository's nested SubsetDataset flattens
    # indices without composing them and cannot wrap this split safely.
    return Subset(train_dataset, retained), Subset(train_dataset, validation)


def save_manifest(conf, clients, validation_dataset, test_dataset):
    from src.datasets.data_splits import split_mode_to_matrix
    target = np.asarray(split_mode_to_matrix(conf))
    cache = {}
    manifest = {"protocol": "controlled-validation-v1", "data_seed": conf["seed"],
                "validation_seed": options(conf)["validation_seed"],
                "training_seed": options(conf)["training_seed"], "clients": [],
                "validation": metadata(validation_dataset, cache), "test": metadata(test_dataset, cache)}
    seen = set()
    for cid, dataset in enumerate(clients):
        rows = metadata(dataset, cache)
        counts = np.zeros(target[cid].shape, dtype=int)
        for identity, y, g in rows:
            if identity in seen:
                raise ValueError(f"Duplicated training sample: {identity}")
            seen.add(identity)
            counts[y, g] += 1
        if not np.array_equal(counts, target[cid]):
            raise ValueError(f"Client {cid} counts differ from requested split: {counts.tolist()}")
        manifest["clients"].append({"cid": cid, "samples": rows, "counts": counts.tolist()})
    val_ids = {r[0] for r in manifest["validation"]}
    test_ids = {r[0] for r in manifest["test"]}
    if seen & val_ids or seen & test_ids or val_ids & test_ids:
        raise ValueError("Training/validation/test identities overlap")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    # Training seed is kept outside the data identity digest, to compare
    # identical data across independent training repeats.
    data_only = {k: v for k, v in manifest.items() if k != "training_seed"}
    digest = hashlib.sha256(json.dumps(data_only, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = Path("checkpoints") / conf["exp_id"]
    (directory / "data_manifest.json").write_bytes(encoded)
    (directory / "data_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest
