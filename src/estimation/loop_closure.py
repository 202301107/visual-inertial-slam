"""Loop-closure candidates from marker re-observations.

A *visit* is an unbroken stretch of keyframes in which one marker is seen. If the same marker is seen
again in a later visit that starts at least `min_gap_s` after the earlier visit ended (i.e. after a
circuit away from it) the two sightings close a loop. Because both sightings see the SAME physical
marker, the relative camera pose between the two keyframes is known directly from the two
marker->camera poses, without any map:

    T_ci_cj = T_cm(i) @ inv(T_cm(j))        (pose of camera j expressed in camera i's frame)

which is added to the pose graph as an explicit metric relative-pose edge (RelPoseEdge).
`relabel_revisits` gives later visits their own landmark id so a graph can be solved WITHOUT the
implicit loop closure that a shared marker pose would otherwise provide (the control experiment).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.estimation.pose_graph import MarkerNoise, MarkerObs, RelPoseEdge
from src.geometry import inv_T, make_T


@dataclass
class LoopCandidate:
    marker: int
    kf_i: int
    kf_j: int
    visit_a: int
    visit_b: int
    gap_s: float
    edge: RelPoseEdge


def split_visits(obs: list[MarkerObs], kf_t_ns: np.ndarray, visit_gap_s: float = 2.0):
    """marker id -> list of visits, each a list of MarkerObs sorted by keyframe."""
    per = defaultdict(list)
    for o in sorted(obs, key=lambda o: o.kf):
        per[o.marker].append(o)
    out = {}
    for m, lst in per.items():
        visits, cur = [], [lst[0]]
        for o in lst[1:]:
            if (kf_t_ns[o.kf] - kf_t_ns[cur[-1].kf]) * 1e-9 > visit_gap_s:
                visits.append(cur)
                cur = []
            cur.append(o)
        visits.append(cur)
        out[m] = visits
    return out


def find_loop_candidates(obs: list[MarkerObs], kf_t_ns: np.ndarray, min_gap_s: float = 10.0,
                         visit_gap_s: float = 2.0, max_pairs: int = 4,
                         noise: MarkerNoise | None = None) -> list[LoopCandidate]:
    noise = noise or MarkerNoise()
    cands: list[LoopCandidate] = []
    for m, visits in split_visits(obs, kf_t_ns, visit_gap_s).items():
        for a in range(len(visits) - 1):
            b = a + 1
            gap = (kf_t_ns[visits[b][0].kf] - kf_t_ns[visits[a][-1].kf]) * 1e-9
            if gap < min_gap_s:
                continue
            pairs = sorted(((oi.dist + oj.dist, oi, oj) for oi in visits[a] for oj in visits[b]), key=lambda x: x[0])
            used_i, used_j = set(), set()
            for _, oi, oj in pairs:
                if len([c for c in cands if c.marker == m and c.visit_a == a]) >= max_pairs:
                    break
                if oi.kf in used_i or oj.kf in used_j:
                    continue
                used_i.add(oi.kf)
                used_j.add(oj.kf)
                T = make_T(oi.R, oi.t) @ inv_T(make_T(oj.R, oj.t))
                st = np.hypot(noise.t_base + noise.t_per_m * oi.dist, noise.t_base + noise.t_per_m * oj.dist)
                sr = np.hypot(noise.r_base + noise.r_per_m * oi.dist, noise.r_base + noise.r_per_m * oj.dist)
                cands.append(LoopCandidate(m, oi.kf, oj.kf, a, b, float(gap),
                                           RelPoseEdge(oi.kf, oj.kf, T[:3, :3], T[:3, 3], float(sr), float(st))))
    return cands


def relabel_revisits(obs: list[MarkerObs], kf_t_ns: np.ndarray, min_gap_s: float = 10.0,
                     visit_gap_s: float = 2.0, copy_offset: int = 100):
    """Give each marker visit that follows a loop-length gap its own landmark id (marker + k*copy_offset).

    Returns (new obs list, {copy id: original id}). Visits close in time keep one id, so only genuine
    loop revisits are separated.
    """
    new, copies = [], {}
    for m, visits in split_visits(obs, kf_t_ns, visit_gap_s).items():
        k = 0
        for a, v in enumerate(visits):
            if a > 0 and (kf_t_ns[v[0].kf] - kf_t_ns[visits[a - 1][-1].kf]) * 1e-9 >= min_gap_s:
                k += 1
            mid = m + k * copy_offset
            if k:
                copies[mid] = m
            new += [MarkerObs(o.kf, mid, o.R, o.t, o.dist) for o in v]
    return new, copies
