"""Shared fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture()
def aml_data_dir(tmp_path):
    """A tiny fake preprocessed AMLworld ``graph.pt`` (500 time-sorted edges, splits 300/100/100)."""
    rng = np.random.default_rng(0)
    n, sizes = 60, (300, 100, 100)
    e = sum(sizes)
    ei = torch.from_numpy(rng.integers(0, n, size=(2, e)))
    t = torch.from_numpy(np.sort(rng.integers(0, 3 * 86400, size=e)))
    from amlinterp.data.ports import compute_ports
    g = {
        "edge_index": ei, "edge_time": t,
        "amount_received": torch.from_numpy(rng.lognormal(5, 2, e)),
        "receiving_currency": torch.from_numpy(rng.integers(0, 5, e)),
        "payment_format": torch.from_numpy(rng.integers(0, 3, e)),
        "y": torch.from_numpy((rng.random(e) < 0.1).astype(np.int64)),
        "ports": compute_ports(ei, t.double(), n).int(),
        "split": {"train": torch.arange(0, 300), "val": torch.arange(300, 400), "test": torch.arange(400, 500)},
        "num_nodes": n,
    }
    torch.save(g, tmp_path / "graph.pt")
    return tmp_path
