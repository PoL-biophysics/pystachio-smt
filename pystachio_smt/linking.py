"""Frame-to-frame linking and gap closing, both by linear assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def _assignment_cost(dist):
    """Prefer maximum valid match count, then minimum total distance."""
    finite = np.isfinite(dist)
    scale = max(1.0, float(np.max(dist[finite])) if finite.any() else 1.0)
    return np.where(finite, dist / scale, min(dist.shape) + 1.0)


def _validate(spots, params):
    required = ["frame", "x", "y", "intensity"]
    if not set(required).issubset(spots.columns):
        raise ValueError(f"Detection table requires {required}")
    if not spots.index.is_unique:
        raise ValueError("Detection table index must be unique")
    data = spots[required].to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Detection frames, positions and intensities must be finite")
    frames = data[:, 0]
    if (frames < 0).any() or (frames != np.floor(frames)).any():
        raise ValueError("Frame numbers must be nonnegative integers")
    for name in ("d_01_max", "gap_close_dist", "Iratio_01_min", "Iratio_01_max", "max_gap"):
        value = getattr(params, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if params.Iratio_01_min > params.Iratio_01_max:
        raise ValueError("Intensity ratio minimum exceeds maximum")
    if int(params.max_gap) != params.max_gap:
        raise ValueError("max_gap must be an integer")
    return spots.copy()



def link_frames(spots, params):
    """Join spots between consecutive frames.

    Each frame pair is solved as a linear assignment on Euclidean distance, so
    the whole frame is matched jointly rather than greedily. Candidates further
    apart than `d_01_max`, or whose intensities differ implausibly, are
    forbidden. Spots left unlinked become their own single-frame track.
    """
    spots = _validate(spots, params)
    frames = np.sort(spots['frame'].unique())
    ids = {}
    next_id = 1

    lo = params.Iratio_01_min * 0.5
    hi = params.Iratio_01_max * 2.0

    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            continue
        idx_prev = spots.index[spots['frame'] == frames[i - 1]].tolist()
        idx_cur = spots.index[spots['frame'] == frames[i]].tolist()
        if not idx_prev or not idx_cur:
            continue

        dist = cdist(spots.loc[idx_cur, ['x', 'y']].values,
                     spots.loc[idx_prev, ['x', 'y']].values)
        dist[dist > params.d_01_max] = np.inf

        i_prev = spots.loc[idx_prev, 'intensity'].values
        i_cur = spots.loc[idx_cur, 'intensity'].values
        ratio = i_prev[None, :] / (i_cur[:, None] + 1e-9)
        dist[(ratio < lo) | (ratio > hi)] = np.inf

        cost = _assignment_cost(dist)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if np.isinf(dist[r, c]):
                continue
            prev, cur = idx_prev[c], idx_cur[r]
            if ids.get(prev, 0) == 0:
                ids[prev] = next_id
                next_id += 1
            ids[cur] = ids[prev]

    spots['traj_id'] = np.array([ids.get(ix, 0) for ix in spots.index], dtype=np.int64)
    lone = spots['traj_id'] == 0
    spots.loc[lone, 'traj_id'] = np.arange(next_id, next_id + lone.sum())
    return spots


def close_gaps(spots, params):
    """Rejoin track segments separated by a few missed frames.

    A second assignment, this time between segment ends and segment starts.
    Segments are merged through a union-find so a chain of joins collapses to a
    single track.
    """
    spots = _validate(spots, params)
    if "traj_id" not in spots or spots["traj_id"].isna().any():
        raise ValueError("close_gaps requires valid traj_id values from link_frames")
    if spots.duplicated(["traj_id", "frame"]).any():
        raise ValueError("A trajectory cannot contain two detections in one frame")
    max_gap = int(params.max_gap)
    if max_gap < 1:
        return spots

    ids, starts, ends = [], [], []
    for tid, g in spots.sort_values('frame').groupby('traj_id'):
        ids.append(tid)
        starts.append((g['frame'].iloc[0], g['x'].iloc[0], g['y'].iloc[0]))
        ends.append((g['frame'].iloc[-1], g['x'].iloc[-1], g['y'].iloc[-1]))
    n = len(ids)
    if n < 2:
        return spots

    ids = np.array(ids)
    starts = np.array(starts)
    ends = np.array(ends)

    cost = np.full((n, n), np.inf)
    for i in range(n):
        end_f, ex, ey = ends[i]
        gap = starts[:, 0] - end_f
        dist = np.hypot(starts[:, 1] - ex, starts[:, 2] - ey)
        ok = (gap >= 1) & (gap <= max_gap) & (dist <= params.gap_close_dist)
        ok[i] = False
        # Tiny gap term only breaks ties between equidistant candidates.
        cost[i, ok] = dist[ok] + (gap[ok] - 1) * 1e-3

    if not np.isfinite(cost).any():
        return spots

    rows, cols = linear_sum_assignment(_assignment_cost(cost))

    parent = {t: t for t in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r, c in zip(rows, cols):
        if not np.isfinite(cost[r, c]):
            continue
        ra, rb = find(ids[r]), find(ids[c])
        if ra != rb:
            parent[rb] = ra

    spots['traj_id'] = spots['traj_id'].map(find)
    relabel = {t: i + 1 for i, t in enumerate(sorted(spots['traj_id'].unique()))}
    spots['traj_id'] = spots['traj_id'].map(relabel)
    return spots
