from __future__ import annotations

"""Lightweight Python implementation of the DSurfTomo workflow.

This module mirrors the input/output contract of the original Fortran driver
while favouring readability and portability.  The inversion logic is
purposefully simplified but retains the same file formats so existing data
preparation and post-processing scripts continue to work.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


EARTH_RADIUS_KM = 6371.0


@dataclass
class SourceMeasurement:
    """Container for measurements associated with one source and period."""

    latitude_deg: float
    longitude_deg: float
    period: int
    wave_type: int
    velocity_type: int
    receivers: List[Tuple[float, float, float]]  # lat, lon, phase/group velocity

    @property
    def colat_rad(self) -> float:
        # Fortran converts latitude to colatitude in radians: (90 - lat) * pi/180
        return np.deg2rad(90.0 - self.latitude_deg)

    @property
    def lon_rad(self) -> float:
        return np.deg2rad(self.longitude_deg)


@dataclass
class InputConfig:
    datafile: Path
    nx: int
    ny: int
    nz: int
    goxd: float
    gozd: float
    dvxd: float
    dvzd: float
    nsrc: int
    weight0: float
    damp: float
    minthk: float
    min_vel: float
    max_vel: float
    maxiter: int
    spfra: float
    kmax_rc: int
    t_rc: List[float]
    kmax_rg: int
    t_rg: List[float]
    kmax_lc: int
    t_lc: List[float]
    kmax_lg: int
    t_lg: List[float]
    ifsyn: int
    noiselevel: float
    threshold0: float


@dataclass
class Model:
    depz: np.ndarray
    vsf: np.ndarray

    @property
    def interior(self) -> np.ndarray:
        # Interior nodes match the original Fortran usage (skip boundaries)
        return self.vsf[1:-1, 1:-1, :-1]


@dataclass
class ObservationSet:
    distances: np.ndarray
    observed_times: np.ndarray
    datweight: np.ndarray



def parse_input_file(path: Path) -> InputConfig:
    lines = path.read_text().splitlines()
    # First three lines are descriptive headers in the original Fortran code
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
    kmax_rc = int(lines[cursor].split()[0]); cursor += 1
    t_rc = []
    if kmax_rc > 0:
        t_rc = list(map(float, lines[cursor].split()))
        cursor += 1
    kmax_rg = int(lines[cursor].split()[0]); cursor += 1
    t_rg = []
    if kmax_rg > 0:
        t_rg = list(map(float, lines[cursor].split()))
        cursor += 1
    kmax_lc = int(lines[cursor].split()[0]); cursor += 1
    t_lc = []
    if kmax_lc > 0:
        t_lc = list(map(float, lines[cursor].split()))
        cursor += 1
    kmax_lg = int(lines[cursor].split()[0]); cursor += 1
    t_lg = []
    if kmax_lg > 0:
        t_lg = list(map(float, lines[cursor].split()))
        cursor += 1
    ifsyn = int(lines[cursor].split()[0]); cursor += 1
    noiselevel = float(lines[cursor].split()[0]); cursor += 1
    threshold0 = float(lines[cursor].split()[0]);

    return InputConfig(
        datafile=datafile,
        nx=nx,
        ny=ny,
        nz=nz,
        goxd=goxd,
        gozd=gozd,
        dvxd=dvxd,
        dvzd=dvzd,
        nsrc=nsrc,
        weight0=weight0,
        damp=damp,
        minthk=minthk,
        min_vel=min_vel,
        max_vel=max_vel,
        maxiter=maxiter,
        spfra=spfra,
        kmax_rc=kmax_rc,
        t_rc=t_rc,
        kmax_rg=kmax_rg,
        t_rg=t_rg,
        kmax_lc=kmax_lc,
        t_lc=t_lc,
        kmax_lg=kmax_lg,
        t_lg=t_lg,
        ifsyn=ifsyn,
        noiselevel=noiselevel,
        threshold0=threshold0,
    )



def great_circle_distance(colat1: float, lon1: float, colat2: float, lon2: float) -> float:
    dlat = colat2 - colat1
    dlon = lon2 - lon1
    lat1 = np.pi / 2 - colat1
    lat2 = np.pi / 2 - colat2
    a = np.sin(dlat / 2) ** 2 + np.sin(dlon / 2) ** 2 * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
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
            colat1 = current.colat_rad
            lon1 = current.lon_rad
            colat2 = np.deg2rad(90.0 - lat2)
            lon2 = np.deg2rad(lon2)
            dist = great_circle_distance(colat1, lon1, colat2, lon2)
            distances.append(dist)
            observed_times.append(dist / velvalue)
    if current is not None:
        sources.append(current)

    return sources, np.array(distances, dtype=float), np.array(observed_times, dtype=float)



def load_model(path: Path, nx: int, ny: int, nz: int) -> Model:
    with path.open("r") as fh:
        depz = np.array(list(map(float, fh.readline().split())), dtype=float)
        vsf = np.zeros((nx, ny, nz), dtype=float)
        for k in range(nz):
            for j in range(ny):
                row = fh.readline()
                vsf[:, j, k] = np.array(list(map(float, row.split())), dtype=float)
    return Model(depz=depz, vsf=vsf)



def write_model(filepath: Path, model: Model, gozd: float, goxd: float, dvzd: float, dvxd: float) -> None:
    with filepath.open("w") as fh:
        nx, ny, nz = model.vsf.shape
        for k in range(nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    fh.write(
                        f"{gozd + (j-1)*dvzd:10.5f}{goxd - (i-1)*dvxd:10.5f}{model.depz[k]:10.5f}{model.vsf[i, j, k]:10.5f}\n"
                    )



def percentile_bounds(values: np.ndarray, lower: float = 25.0, upper: float = 75.0) -> Tuple[float, float]:
    q25 = np.percentile(values, lower)
    q75 = np.percentile(values, upper)
    return float(q25), float(q75)



def synthetic_noise(residual: np.ndarray, level: float) -> np.ndarray:
    if level <= 0:
        return residual
    rng = np.random.default_rng(12345)
    return residual + rng.normal(scale=level, size=residual.shape)



def update_velocities(model: Model, scale: float, min_vel: float, max_vel: float) -> None:
    interior = model.interior
    interior *= scale
    np.clip(interior, min_vel, max_vel, out=interior)
    model.vsf[1:-1, 1:-1, :-1] = interior



def compute_predictions(distances: np.ndarray, model: Model) -> np.ndarray:
    mean_vel = np.mean(model.interior)
    mean_vel = max(mean_vel, 1e-6)
    return distances / mean_vel



def run_iteration(
    model: Model, observation: ObservationSet, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    synthetic = compute_predictions(observation.distances, model)
    residual = observation.observed_times - synthetic
    q25, q75 = percentile_bounds(residual)
    datweight = np.ones_like(residual)
    mask = (residual < q25 * threshold) | (residual > q75 * threshold)
    datweight[mask] = 0.0
    weighted_residual = residual * datweight
    rms = np.linalg.norm(weighted_residual) / np.sqrt(max(len(weighted_residual), 1))
    return synthetic, datweight, weighted_residual, float(np.mean(weighted_residual)), float(rms)



def solve_scale_factor(distances: np.ndarray, observed_times: np.ndarray, current_model: Model) -> float:
    # Simple closed-form scale factor that matches mean observed velocity.
    target_vel = np.sum(distances) / np.sum(observed_times)
    current_vel = np.mean(current_model.interior)
    if current_vel <= 0:
        return 1.0
    return target_vel / current_vel



def write_residuals(path: Path, distances: np.ndarray, synthetic: np.ndarray, observed: np.ndarray, datweight: np.ndarray) -> None:
    with path.open("w") as fh:
        for d, s, o, w in zip(distances, synthetic, observed, datweight):
            fh.write(f"{d:12.5f}{s:12.5f}{o:12.5f}{s*w:12.5f}{o*w:12.5f}{w:10.3f}\n")



def log_header(log_path: Path, cfg: InputConfig) -> None:
    with log_path.open("w") as log:
        log.write("                         S U R F  T O M O\n")
        log.write("PLEASE contact Hongjain Fang (fanghj1990@gmail.com) if you find any bug\n\n")
        log.write("model origin:latitude,longitue\n")
        log.write(f"{cfg.goxd:10.5f}{cfg.gozd:10.5f}\n")
        log.write("grid spacing:latitude,longitue\n")
        log.write(f"{cfg.dvxd:10.5f}{cfg.dvzd:10.5f}\n")
        log.write("model dimension:nx,ny,nz\n")
        log.write(f"{cfg.nx:5d}{cfg.ny:5d}{cfg.nz:5d}\n")



def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Python implementation of DSurfTomo")
    parser.add_argument("inputfile", nargs="?", default="DSurfTomo.in", help="Input configuration file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.inputfile)
    if not input_path.exists():
        raise SystemExit(f"unable to open the inputfile {input_path}")

    cfg = parse_input_file(input_path)
    sources, distances, observed_times = parse_measurements(cfg.datafile)
    model = load_model(Path("MOD"), cfg.nx, cfg.ny, cfg.nz)

    log_path = input_path.with_suffix(input_path.suffix + ".log")
    log_header(log_path, cfg)

    observation = ObservationSet(distances=distances, observed_times=observed_times, datweight=np.ones_like(observed_times))

    for iteration in range(1, cfg.maxiter + 1):
        synthetic, datweight, weighted_residual, mean_res, rms = run_iteration(model, observation, cfg.threshold0)
        observation.datweight = datweight

        if iteration == 1:
            write_residuals(Path("residualFirst.dat"), distances, synthetic, observed_times, observation.datweight)
        if iteration == cfg.maxiter:
            write_residuals(Path("residualLast.dat"), distances, synthetic, observed_times, observation.datweight)

        scale = solve_scale_factor(distances, observed_times, model)
        update_velocities(model, scale, cfg.min_vel, cfg.max_vel)

        outmodel = Path(f"{input_path.name}Measure.dat.iter{iteration:03d}")
        write_model(outmodel, model, cfg.gozd, cfg.goxd, cfg.dvzd, cfg.dvxd)

        with log_path.open("a") as log:
            log.write(f"{iteration:2d}th iteration...\n")
            log.write(f"mean,std_devs and rms of residual: {mean_res*1000:8.1f}ms {np.std(weighted_residual)*1000:8.2f}ms {rms:8.3f}\n")
            log.write(f"min and max velocity variation {model.interior.min():7.4f} {model.interior.max():7.4f}\n")

    if cfg.ifsyn == 1:
        with Path("Vs_model.real").open("w") as fh:
            write_model(Path("Vs_model.real"), model, cfg.gozd, cfg.goxd, cfg.dvzd, cfg.dvxd)
        outsyn = Path(f"{input_path.name}Syn.dat")
        write_model(outsyn, model, cfg.gozd, cfg.goxd, cfg.dvzd, cfg.dvxd)
    else:
        outmodel = Path(f"{input_path.name}Measure.dat")
        write_model(outmodel, model, cfg.gozd, cfg.goxd, cfg.dvzd, cfg.dvxd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
