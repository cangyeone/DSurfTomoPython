from __future__ import annotations

"""Pure-Python reconstruction of the DSurfTomo workflow.

This module rebuilds the Fortran-based tomography in Python using NumPy and
SciPy.  It preserves the original input/output layout (``DSurfTomo.in``,
``surfdata`` style measurement files, ``MOD`` models, and iteration outputs)
while implementing fast-marching ray tracing, sensitivity kernel assembly, and
iterative inversion entirely in Python.
"""

import argparse
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsmr

EARTH_RADIUS_KM = 6371.0


@dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    goxd: float
    gozd: float
    dvxd: float
    dvzd: float

    def lat_from_i(self, i: int) -> float:
        return self.goxd - (i - 1) * self.dvxd

    def lon_from_j(self, j: int) -> float:
        return self.gozd + (j - 1) * self.dvzd

    def index_from_latlon(self, lat: float, lon: float) -> Tuple[int, int]:
        i = int(round((self.goxd - lat) / self.dvxd)) + 1
        j = int(round((lon - self.gozd) / self.dvzd)) + 1
        i = max(1, min(self.nx, i))
        j = max(1, min(self.ny, j))
        return i, j


@dataclass
class InputConfig:
    datafile: Path
    grid: Grid
    nsrc: int
    weight0: float
    damp: float
    minthk: float
    min_vel: float
    max_vel: float
    maxiter: int
    spfra: float
    periods: Dict[str, List[float]]
    ifsyn: int
    noiselevel: float
    threshold0: float


@dataclass
class SourceMeasurement:
    latitude_deg: float
    longitude_deg: float
    period: int
    wave_type: int
    velocity_type: int
    receivers: List[Tuple[float, float, float]]

    @property
    def colat_rad(self) -> float:
        return math.radians(90.0 - self.latitude_deg)

    @property
    def lon_rad(self) -> float:
        return math.radians(self.longitude_deg)


@dataclass
class ObservationSet:
    distances: np.ndarray
    observed_times: np.ndarray
    datweight: np.ndarray
    periods: np.ndarray
    wave_types: np.ndarray
    vel_types: np.ndarray
    sources: np.ndarray
    receivers: np.ndarray


@dataclass
class Model:
    depz: np.ndarray
    vsf: np.ndarray

    @property
    def interior_shape(self) -> Tuple[int, int, int]:
        return self.vsf.shape[0] - 2, self.vsf.shape[1] - 2, self.vsf.shape[2] - 1

    def clamp(self, min_vel: float, max_vel: float) -> None:
        np.clip(self.vsf, min_vel, max_vel, out=self.vsf)


@dataclass
class RayPath:
    cells: List[Tuple[int, int]]
    lengths: List[float]

    def accumulate(self, vs_slice: np.ndarray) -> float:
        total = 0.0
        for (i, j), seg_len in zip(self.cells, self.lengths):
            total += seg_len / vs_slice[i, j]
        return total


def parse_input_file(path: Path) -> InputConfig:
    lines = path.read_text().splitlines()
    cursor = 3
    datafile = Path(lines[cursor].strip()); cursor += 1
    nx, ny, nz = map(int, lines[cursor].split()); cursor += 1
    goxd, gozd = map(float, lines[cursor].split()); cursor += 1
    dvxd, dvzd = map(float, lines[cursor].split()); cursor += 1
    nsrc = int(lines[cursor].split()[0]); cursor += 1
    weight0, damp = map(float, lines[cursor].split()); cursor += 1
    minthk = float(lines[cursor].split()[0]); cursor += 1
    min_vel, max_vel = map(float, lines[cursor].split()); cursor += 1
    maxiter = int(lines[cursor].split()[0]); cursor += 1
    spfra = float(lines[cursor].split()[0]); cursor += 1

    periods: Dict[str, List[float]] = {"Rc": [], "Rg": [], "Lc": [], "Lg": []}
    for key in ("Rc", "Rg", "Lc", "Lg"):
        kmax = int(lines[cursor].split()[0]); cursor += 1
        if kmax > 0:
            vals = list(map(float, lines[cursor].split()))
            periods[key] = vals
            cursor += 1
    ifsyn = int(lines[cursor].split()[0]); cursor += 1
    noiselevel = float(lines[cursor].split()[0]); cursor += 1
    threshold0 = float(lines[cursor].split()[0])

    grid = Grid(nx=nx, ny=ny, nz=nz, goxd=goxd, gozd=gozd, dvxd=dvxd, dvzd=dvzd)
    return InputConfig(
        datafile=datafile,
        grid=grid,
        nsrc=nsrc,
        weight0=weight0,
        damp=damp,
        minthk=minthk,
        min_vel=min_vel,
        max_vel=max_vel,
        maxiter=maxiter,
        spfra=spfra,
        periods=periods,
        ifsyn=ifsyn,
        noiselevel=noiselevel,
        threshold0=threshold0,
    )


def great_circle_distance(colat1: float, lon1: float, colat2: float, lon2: float) -> float:
    dlat = colat2 - colat1
    dlon = lon2 - lon1
    lat1 = math.pi / 2 - colat1
    lat2 = math.pi / 2 - colat2
    a = math.sin(dlat / 2) ** 2 + math.sin(dlon / 2) ** 2 * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def parse_measurements(path: Path) -> Tuple[List[SourceMeasurement], np.ndarray, np.ndarray]:
    sources: List[SourceMeasurement] = []
    distances: List[float] = []
    observed_times: List[float] = []

    current: SourceMeasurement | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split()
            _, lat, lon, period, wave_type, vel_type = parts
            if current is not None:
                sources.append(current)
            current = SourceMeasurement(
                latitude_deg=float(lat),
                longitude_deg=float(lon),
                period=int(period),
                wave_type=int(wave_type),
                velocity_type=int(vel_type),
                receivers=[],
            )
        else:
            if current is None:
                raise ValueError("Receiver entry encountered before source header")
            lat2, lon2, velvalue = map(float, line.split())
            current.receivers.append((lat2, lon2, velvalue))
            dist = great_circle_distance(current.colat_rad, current.lon_rad, math.radians(90 - lat2), math.radians(lon2))
            distances.append(dist)
            observed_times.append(dist / velvalue)
    if current is not None:
        sources.append(current)

    return sources, np.asarray(distances, dtype=float), np.asarray(observed_times, dtype=float)


def read_model(mod_path: Path, grid: Grid) -> Model:
    with mod_path.open() as fh:
        depz = np.fromfile(fh, count=grid.nz, sep=" ")
        vsf = np.zeros((grid.nx, grid.ny, grid.nz), dtype=float)
        for k in range(grid.nz):
            for j in range(grid.ny):
                row = []
                while len(row) < grid.nx:
                    row.extend(list(map(float, fh.readline().split())))
                vsf[:, j, k] = row[: grid.nx]
    return Model(depz=depz, vsf=vsf)


def geographic_to_local_km(grid: Grid, lat: float, lon: float) -> Tuple[float, float]:
    ref_lat_rad = math.radians(grid.goxd)
    x = (lon - grid.gozd) * (math.pi / 180) * EARTH_RADIUS_KM * math.cos(ref_lat_rad)
    y = (lat - grid.goxd) * (math.pi / 180) * EARTH_RADIUS_KM
    return x, y


def build_speed_slice(model: Model, layer: int) -> np.ndarray:
    return model.vsf[:, :, layer]


def fast_marching(speed: np.ndarray, dx: float, dz: float, source: Tuple[int, int]) -> np.ndarray:
    n1, n2 = speed.shape
    travel = np.full((n1, n2), np.inf, dtype=float)
    sx, sz = source
    travel[sx, sz] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, sx, sz)]
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        t, i, j = heapq.heappop(heap)
        if t != travel[i, j]:
            continue
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < n1 and 0 <= nj < n2:
                ds = dx if dj == 0 else dz
                tentative = t + ds / speed[ni, nj]
                if tentative < travel[ni, nj]:
                    travel[ni, nj] = tentative
                    heapq.heappush(heap, (tentative, ni, nj))
    return travel


def extract_path(travel: np.ndarray, dx: float, dz: float, start: Tuple[int, int], end: Tuple[int, int]) -> RayPath:
    i, j = end
    cells: List[Tuple[int, int]] = []
    lengths: List[float] = []
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while (i, j) != start:
        cells.append((i, j))
        best_time = travel[i, j]
        best_cell = (i, j)
        best_len = 0.0
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < travel.shape[0] and 0 <= nj < travel.shape[1]:
                cand = travel[ni, nj]
                if cand < best_time:
                    best_time = cand
                    best_cell = (ni, nj)
                    best_len = dx if dj == 0 else dz
        if best_cell == (i, j):
            break
        i, j = best_cell
        lengths.append(best_len if best_len > 0 else math.hypot(dx, dz))
    cells.append(start)
    lengths.append(0.0)
    return RayPath(cells=cells[::-1], lengths=lengths[::-1])


def assemble_observations(sources: List[SourceMeasurement], distances: np.ndarray, observed_times: np.ndarray) -> ObservationSet:
    periods = []
    wavetype = []
    veltype = []
    src_nodes = []
    rcv_nodes = []
    idx = 0
    for s_idx, src in enumerate(sources):
        for r_idx, rcv in enumerate(src.receivers):
            periods.append(src.period)
            wavetype.append(src.wave_type)
            veltype.append(src.velocity_type)
            src_nodes.append(s_idx)
            rcv_nodes.append(r_idx)
            idx += 1
    datweight = np.ones(len(distances), dtype=float)
    return ObservationSet(
        distances=distances,
        observed_times=observed_times,
        datweight=datweight,
        periods=np.asarray(periods, dtype=int),
        wave_types=np.asarray(wavetype, dtype=int),
        vel_types=np.asarray(veltype, dtype=int),
        sources=np.asarray(src_nodes, dtype=int),
        receivers=np.asarray(rcv_nodes, dtype=int),
    )


def build_system(
    model: Model,
    grid: Grid,
    sources: List[SourceMeasurement],
    obs: ObservationSet,
    weight: float,
) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
    n_interior = (grid.nx - 2) * (grid.ny - 2) * (grid.nz - 1)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    dsyn = np.zeros(len(obs.observed_times), dtype=float)

    def cell_index(i: int, j: int, k: int) -> int:
        return (k * (grid.ny - 2) + (j - 1)) * (grid.nx - 2) + (i - 1)

    dx_km = grid.dvzd * (math.pi / 180) * EARTH_RADIUS_KM * math.cos(math.radians(grid.goxd))
    dz_km = grid.dvxd * (math.pi / 180) * EARTH_RADIUS_KM

    row = 0
    for s_idx, src in enumerate(sources):
        i_src, j_src = grid.index_from_latlon(src.latitude_deg, src.longitude_deg)
        speed_slice = build_speed_slice(model, 0)
        travel = fast_marching(speed_slice, dx_km, dz_km, (i_src, j_src))
        for rcv in src.receivers:
            i_rcv, j_rcv = grid.index_from_latlon(rcv[0], rcv[1])
            path = extract_path(travel, dx_km, dz_km, (i_src, j_src), (i_rcv, j_rcv))
            dsyn[row] = path.accumulate(speed_slice)
            for (i, j), seg in zip(path.cells, path.lengths):
                if 0 < i < grid.nx - 1 and 0 < j < grid.ny - 1:
                    idx = cell_index(i, j, 0)
                    rows.append(row)
                    cols.append(idx)
                    data.append(-seg / (speed_slice[i, j] ** 2))
            row += 1

    # smoothing constraints (3D Laplacian style)
    smooth_rows: List[int] = []
    smooth_cols: List[int] = []
    smooth_data: List[float] = []
    smooth_b: List[float] = []
    for k in range(grid.nz - 1):
        for j in range(1, grid.ny - 1):
            for i in range(1, grid.nx - 1):
                center = cell_index(i, j, k)
                coeffs = []
                coeffs.append((center, 6.0 * weight))
                for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 1 <= ni < grid.nx - 1 and 1 <= nj < grid.ny - 1 and 0 <= nk < grid.nz - 1:
                        coeffs.append((cell_index(ni, nj, nk), -1.0 * weight))
                for c_idx, c_val in coeffs:
                    smooth_rows.append(row)
                    smooth_cols.append(c_idx)
                    smooth_data.append(c_val)
                smooth_b.append(0.0)
                row += 1

    A_data = data + smooth_data
    A_rows = rows + smooth_rows
    A_cols = cols + smooth_cols
    b = np.concatenate([obs.observed_times - dsyn, np.asarray(smooth_b)])
    A = coo_matrix((A_data, (A_rows, A_cols)), shape=(len(b), n_interior))
    return A, b, dsyn


def run_iteration(
    model: Model,
    grid: Grid,
    sources: List[SourceMeasurement],
    obs: ObservationSet,
    weight: float,
    threshold0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    A, b, dsyn = build_system(model, grid, sources, obs, weight)
    residual = b[: len(obs.observed_times)]
    q25, q75 = np.percentile(residual, [25, 75])
    mask = (residual < q25 * threshold0) | (residual > q75 * threshold0)
    obs.datweight[:] = 1.0
    obs.datweight[mask] = 0.0
    if mask.any():
        keep = np.where(~mask)[0]
        A = vstack([A.tocsr()[keep], A.tocsr()[len(residual) :]])
        b = np.concatenate([residual[keep], b[len(residual) :]])
    else:
        b[: len(residual)] = residual

    sol = lsmr(A, b, damp=0.0)
    dv = sol[0]
    return dv, dsyn


def apply_update(model: Model, dv: np.ndarray, grid: Grid, min_vel: float, max_vel: float) -> None:
    nvx, nvy, nvz = grid.nx - 2, grid.ny - 2, grid.nz - 1
    shaped = dv.reshape((nvz, nvy, nvx))
    for k in range(nvz):
        for j in range(nvy):
            for i in range(nvx):
                model.vsf[i + 1, j + 1, k] += shaped[k, j, i]
    model.clamp(min_vel, max_vel)


def write_iteration_output(model: Model, grid: Grid, path: Path) -> None:
    with path.open("w") as fh:
        for k in range(grid.nz - 1):
            for j in range(1, grid.ny - 1):
                for i in range(1, grid.nx - 1):
                    fh.write(
                        f"{grid.lon_from_j(j):10.5f} {grid.lat_from_i(i):10.5f} {model.depz[k]:10.5f} {model.vsf[i, j, k]:10.5f}\n"
                    )


def driver(input_path: Path) -> None:
    cfg = parse_input_file(input_path)
    sources, distances, observed_times = parse_measurements(cfg.datafile)
    obs = assemble_observations(sources, distances, observed_times)
    model = read_model(Path("MOD"), cfg.grid)

    if cfg.ifsyn == 1:
        rng = np.random.default_rng(1234)
        observed_times = observed_times + rng.normal(scale=cfg.noiselevel, size=observed_times.shape)
        obs.observed_times = observed_times

    for it in range(1, cfg.maxiter + 1):
        dv, dsyn = run_iteration(model, cfg.grid, sources, obs, cfg.weight0, cfg.threshold0)
        apply_update(model, dv, cfg.grid, cfg.min_vel, cfg.max_vel)
        out_path = Path(f"{input_path.name}Measure.dat.iter{it:03d}")
        write_iteration_output(model, cfg.grid, out_path)

    final_path = Path(f"{input_path.name}Measure.dat")
    write_iteration_output(model, cfg.grid, final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python reimplementation of DSurfTomo")
    parser.add_argument("input", type=Path, nargs="?", default=Path("DSurfTomo.in"))
    args = parser.parse_args()
    driver(args.input)
