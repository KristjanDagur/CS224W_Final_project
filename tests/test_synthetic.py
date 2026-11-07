"""Check the synthetic labelers against brute-force reference implementations on small graphs."""

from itertools import combinations, permutations

import networkx as nx
import numpy as np
import pytest

from amlinterp.data.synthetic import (
    LABEL_NAMES,
    CirculantConfig,
    _simple_adjacency,
    biclique_labels,
    compute_labels,
    cycle_labels,
    degree_fan_labels,
    generate_circulant_graph,
    scatter_gather_labels,
)

SMALL = CirculantConfig(n=120, d=6.0, r=3.0)  # dense enough that every pattern occurs


def _small_graph(seed: int) -> np.ndarray:
    return generate_circulant_graph(SMALL, np.random.default_rng(seed))[0]


def _edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {(int(u), int(v)) for u, v in edge_index.T if u != v}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cycles_match_networkx(seed):
    ei = _small_graph(seed)
    g = nx.DiGraph(list(_edge_set(ei)))
    expected = np.zeros((SMALL.n, 5), dtype=bool)
    for cyc in nx.simple_cycles(g, length_bound=6):
        if len(cyc) >= 2:
            expected[cyc, len(cyc) - 2] = True
    got = cycle_labels(_simple_adjacency(ei, SMALL.n))
    np.testing.assert_array_equal(got, expected)
    assert got.any(axis=0).all(), "every cycle length should occur in this test graph"


@pytest.mark.parametrize("seed", [0, 1])
def test_scatter_gather_and_biclique_brute_force(seed):
    ei = _small_graph(seed)
    edges = _edge_set(ei)
    n = SMALL.n
    out = {u: set() for u in range(n)}
    inn = {v: set() for v in range(n)}
    for u, v in edges:
        out[u].add(v)
        inn[v].add(u)
    sg = np.zeros(n, dtype=bool)
    for v in range(n):
        for s in range(n):
            if s != v and len({w for w in out[s] if v in out[w] and w not in (s, v)}) >= 2:
                sg[v] = True
                break
    bc = np.zeros(n, dtype=bool)
    for v, v2 in permutations(range(n), 2):
        common = (inn[v] & inn[v2]) - {v, v2}
        if any(s1 != s2 for s1, s2 in combinations(common, 2)):
            bc[v] = True
    adj = _simple_adjacency(ei, n)
    np.testing.assert_array_equal(scatter_gather_labels(adj), sg)
    np.testing.assert_array_equal(biclique_labels(adj), bc)
    assert sg.any() and (~sg).any() and bc.any() and (~bc).any()


def test_degree_fan_counts_parallel_edges_and_self_loops():
    # node 0: 4 parallel edges to 1 (deg_out 4, fan_out 1); node 2: self-loop + 3 out-neighbors
    ei = np.array([[0, 0, 0, 0, 2, 2, 2, 2], [1, 1, 1, 1, 2, 3, 4, 5]])
    lab = degree_fan_labels(ei, 6)  # columns: deg_in, deg_out, fan_in, fan_out
    assert lab[0].tolist() == [False, True, False, False]
    assert lab[1].tolist() == [True, False, False, False]
    assert lab[2].tolist() == [False, True, False, True]  # self-loop counts as an out-neighbor


def test_compute_labels_shape_and_order():
    ei = _small_graph(0)
    y = compute_labels(ei, SMALL.n)
    assert y.shape == (SMALL.n, len(LABEL_NAMES)) and y.dtype == bool


def test_generator_size_and_locality():
    cfg = CirculantConfig()
    ei, t = generate_circulant_graph(cfg, np.random.default_rng(0))
    assert ei.shape == (2, int(cfg.n * cfg.d // 2)) and t.shape == (ei.shape[1],)
    assert ei.min() >= 0 and ei.max() < cfg.n
    assert (ei[0] != ei[1]).all(), "self-loops must be re-drawn"
    ring_dist = np.minimum((ei[1] - ei[0]) % cfg.n, (ei[0] - ei[1]) % cfg.n)
    assert np.std(ring_dist) < 2 * cfg.r  # heads stay near tails on the ring
