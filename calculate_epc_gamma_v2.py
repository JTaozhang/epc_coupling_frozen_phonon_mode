#!/usr/bin/env python3
"""
Gamma-point frozen-phonon EPC from TB Hamiltonians.

For tb_hr.dat columns
    i j R1_frac R2_frac R3_frac Re Im

R_frac is treated as the actual i->j displacement vector. Therefore H_plus and
H_minus must be Fourier transformed independently:

    H_plus(k)  = sum t_plus  exp(+i 2*pi*k.R_plus)
    H_minus(k) = sum t_minus exp(+i 2*pi*k.R_minus)

Then, for u_applied = s*u_ZP,

    DeltaH_ep(k) = [H_plus(k)-H_minus(k)]/(2*s)

and using EQUILIBRIUM eigenvectors,

    g_mn(k,Gamma) = <psi_mk^0|DeltaH_ep(k)|psi_nk^0>.

The hopping graph is checked using only (i,j), NOT R_frac.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class HoppingData:
    """Vectorized hopping-list data."""

    i0: np.ndarray
    j0: np.ndarray
    r_frac: np.ndarray
    values: np.ndarray
    nions_inferred: int

    @property
    def nhop(self) -> int:
        """Return number of hopping entries."""
        return int(self.values.size)


@dataclass
class WavefunctionIndex:
    """Byte offsets for selected wavefunction blocks."""

    nions: int
    nkpts_header: int | None
    offsets: dict[tuple[int, int], int]


def read_hopping_list(path: str | Path) -> HoppingData:
    """
    Read tb_hr.dat-like hopping data.

    Parameters
    ----------
    path
        File with columns i j R1_frac R2_frac R3_frac Re Im.

    Returns
    -------
    HoppingData
        Vectorized hopping data.
    """
    ii, jj, rr, vv = [], [], [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            p = line.split()
            if len(p) < 7:
                continue

            try:
                i = int(p[0]) - 1
                j = int(p[1]) - 1
                r = (float(p[2]), float(p[3]), float(p[4]))
                v = float(p[5]) + 1j * float(p[6])
            except ValueError:
                continue

            ii.append(i)
            jj.append(j)
            rr.append(r)
            vv.append(v)

    if not vv:
        raise ValueError(f"No hopping entries found in {path}")

    i0 = np.asarray(ii, dtype=np.int64)
    j0 = np.asarray(jj, dtype=np.int64)
    r_frac = np.asarray(rr, dtype=np.float64)
    values = np.asarray(vv, dtype=np.complex128)

    return HoppingData(
        i0=i0,
        j0=j0,
        r_frac=r_frac,
        values=values,
        nions_inferred=int(max(i0.max(), j0.max()) + 1),
    )


def _packed_pair_keys(hops: HoppingData, base: int) -> np.ndarray:
    """
    Pack (i,j) into one integer key.

    Parameters
    ----------
    hops
        Hopping data.
    base
        Integer larger than all orbital indices.

    Returns
    -------
    numpy.ndarray
        Packed keys.
    """
    return hops.i0 * np.int64(base) + hops.j0


def check_hopping_graph_by_pair(
    plus: HoppingData,
    minus: HoppingData,
    report_path: str | Path,
    sample_count: int = 20,
) -> dict[str, int | float]:
    """
    Compare plus/minus hopping graphs using only (i,j) as identity.

    R_frac is excluded because it changes physically under frozen displacement.

    Parameters
    ----------
    plus, minus
        Plus/minus hopping data.
    report_path
        Output text report.
    sample_count
        Number of exclusive-pair examples written.

    Returns
    -------
    dict
        Summary statistics.
    """
    base = max(plus.nions_inferred, minus.nions_inferred) + 1

    kp_all = _packed_pair_keys(plus, base)
    km_all = _packed_pair_keys(minus, base)

    kp = np.unique(kp_all)
    km = np.unique(km_all)

    common = np.intersect1d(kp, km, assume_unique=True)
    only_plus = np.setdiff1d(kp, km, assume_unique=True)
    only_minus = np.setdiff1d(km, kp, assume_unique=True)

    dup_plus = plus.nhop - kp.size
    dup_minus = minus.nhop - km.size

    mask_plus = np.isin(kp_all, only_plus)
    mask_minus = np.isin(km_all, only_minus)

    amp_plus = np.abs(plus.values[mask_plus])
    amp_minus = np.abs(minus.values[mask_minus])

    stats = {
        "plus_entries": plus.nhop,
        "minus_entries": minus.nhop,
        "plus_unique_pairs": int(kp.size),
        "minus_unique_pairs": int(km.size),
        "common_pairs": int(common.size),
        "only_plus_pairs": int(only_plus.size),
        "only_minus_pairs": int(only_minus.size),
        "duplicate_plus_entries": int(dup_plus),
        "duplicate_minus_entries": int(dup_minus),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Hopping graph comparison using (i,j) only\n")
        f.write("=" * 72 + "\n")
        f.write(f"H_plus entries         : {plus.nhop}\n")
        f.write(f"H_minus entries        : {minus.nhop}\n")
        f.write(f"unique (i,j) plus      : {kp.size}\n")
        f.write(f"unique (i,j) minus     : {km.size}\n")
        f.write(f"common (i,j) pairs     : {common.size}\n")
        f.write(f"(i,j) only in plus     : {only_plus.size}\n")
        f.write(f"(i,j) only in minus    : {only_minus.size}\n")
        f.write(f"duplicate plus entries : {dup_plus}\n")
        f.write(f"duplicate minus entries: {dup_minus}\n")

        ref = max(kp.size, km.size, 1)
        f.write(
            f"only-plus fraction      : {100*only_plus.size/ref:.8f} %\n"
        )
        f.write(
            f"only-minus fraction     : {100*only_minus.size/ref:.8f} %\n"
        )

        f.write("\nExclusive hopping |t| statistics\n")
        f.write("-" * 72 + "\n")

        for label, arr in (("only plus", amp_plus), ("only minus", amp_minus)):
            f.write(f"{label}:\n")
            if arr.size == 0:
                f.write("  no entries\n")
            else:
                f.write(f"  number : {arr.size}\n")
                f.write(f"  min    : {arr.min():.16e}\n")
                f.write(f"  median : {np.median(arr):.16e}\n")
                f.write(f"  mean   : {arr.mean():.16e}\n")
                f.write(f"  max    : {arr.max():.16e}\n")

        f.write("\nSample exclusive pairs\n")
        f.write("-" * 72 + "\n")

        for label, keys in (("only plus", only_plus), ("only minus", only_minus)):
            f.write(f"{label}:\n")
            for key in keys[:sample_count]:
                i0 = int(key // base)
                j0 = int(key % base)
                f.write(f"  i={i0+1:8d} j={j0+1:8d}\n")

    print("Hopping graph comparison")
    print("=" * 68)
    print(f"  plus entries       : {plus.nhop}")
    print(f"  minus entries      : {minus.nhop}")
    print(f"  unique pairs plus  : {kp.size}")
    print(f"  unique pairs minus : {km.size}")
    print(f"  common pairs       : {common.size}")
    print(f"  only plus          : {only_plus.size}")
    print(f"  only minus         : {only_minus.size}")
    print(f"  graph report       : {report_path}")

    return stats


def build_hk_sparse(
    hops: HoppingData,
    k_frac: np.ndarray,
    nions: int,
) -> csr_matrix:
    """
    Build sparse H(k) using exp(+i 2*pi*k.R_frac).

    Parameters
    ----------
    hops
        Hopping data for one structure.
    k_frac
        Fractional reciprocal electronic k point.
    nions
        Basis dimension.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse H(k).
    """
    if hops.nions_inferred > nions:
        raise ValueError(
            f"Hopping list requires {hops.nions_inferred} basis functions, "
            f"but wavefunction nions={nions}."
        )

    phase = np.exp(2j * np.pi * (hops.r_frac @ k_frac))
    data = hops.values * phase

    return coo_matrix(
        (data, (hops.i0, hops.j0)),
        shape=(nions, nions),
        dtype=np.complex128,
    ).tocsr()


def read_kpoints(path: str | Path) -> np.ndarray:
    """
    Read VASP-style reciprocal KPOINTS coordinates.

    Parameters
    ----------
    path
        KPOINTS filename.

    Returns
    -------
    numpy.ndarray
        Fractional reciprocal k points, shape (nkpts,3).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f if x.strip()]

    nkpts = int(lines[1].split()[0])

    coords = []
    for line in lines[3:]:
        if len(coords) >= nkpts:
            break
        p = line.split()
        if len(p) >= 3:
            coords.append([float(p[0]), float(p[1]), float(p[2])])

    if len(coords) != nkpts:
        raise ValueError(
            f"KPOINTS declares {nkpts} points but parsed {len(coords)}."
        )

    return np.asarray(coords, dtype=np.float64)


def read_scale_factor_from_metadata(path: str | Path) -> float:
    """
    Read scale_factor_s from frozen-mode metadata.

    Parameters
    ----------
    path
        Metadata JSON.

    Returns
    -------
    float
        Positive scale factor s.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "scale_factor_s" not in data:
        raise KeyError(f"'scale_factor_s' not found in {path}")

    value = data["scale_factor_s"]
    if value is None:
        raise ValueError("scale_factor_s is None.")

    s = float(value)
    if s <= 0:
        raise ValueError(f"scale_factor_s must be positive, got {s}")

    return s


def index_wavefunction_file(
    path: str | Path,
    selected_bands: list[int],
) -> WavefunctionIndex:
    """
    Index selected wavefunction blocks by byte offset in a single scan.

    This avoids rereading the huge wavefunction file for every k point.

    Parameters
    ----------
    path
        Wavefunction file.
    selected_bands
        Global band indices to index.

    Returns
    -------
    WavefunctionIndex
        Header information and selected block offsets.
    """
    wanted = set(selected_bands)
    nions = None
    nkpts_header = None
    offsets: dict[tuple[int, int], int] = {}

    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break

            s = line.strip()

            if s.startswith(b"# Wavefunctions:"):
                text = s.decode("utf-8", errors="ignore")
                p = text.replace("#", "").replace("=", " ").split()

                if "nions" in p:
                    nions = int(p[p.index("nions") + 1])
                if "nkpts" in p:
                    nkpts_header = int(p[p.index("nkpts") + 1])

            elif s.startswith(b"# kpoint"):
                text = s.decode("utf-8", errors="ignore")
                p = text.split()

                if len(p) < 4:
                    continue

                try:
                    ik = int(p[2])
                    ib = int(p[3])
                except ValueError:
                    continue

                if ib in wanted:
                    offsets[(ik, ib)] = f.tell()

    if nions is None:
        raise ValueError("Could not parse nions from wavefunction header.")

    return WavefunctionIndex(
        nions=nions,
        nkpts_header=nkpts_header,
        offsets=offsets,
    )


def read_selected_wavefunctions_at_k(
    path: str | Path,
    wf_index: WavefunctionIndex,
    k_index: int,
    selected_bands: list[int],
) -> np.ndarray:
    """
    Read selected equilibrium eigenvectors for one k point.

    Parameters
    ----------
    path
        Wavefunction file.
    wf_index
        Byte-offset index.
    k_index
        k index appearing in '# kpoint <ik> <ib>'.
    selected_bands
        Global band indices, in desired column order.

    Returns
    -------
    numpy.ndarray
        Shape (nions,nselected) complex eigenvectors.
    """
    psi = np.empty(
        (wf_index.nions, len(selected_bands)),
        dtype=np.complex128,
    )

    with open(path, "rb") as f:
        for col, band in enumerate(selected_bands):
            key = (k_index, band)

            if key not in wf_index.offsets:
                raise ValueError(
                    f"Missing wavefunction block k={k_index}, band={band}."
                )

            f.seek(wf_index.offsets[key])

            for i in range(wf_index.nions):
                line = f.readline()
                if not line:
                    raise ValueError(
                        f"Unexpected EOF at k={k_index}, band={band}, row={i}."
                    )

                p = line.strip().split()
                if len(p) < 2 or p[0].startswith(b"#"):
                    raise ValueError(
                        f"Malformed/short block at k={k_index}, "
                        f"band={band}, row={i}."
                    )

                psi[i, col] = float(p[0]) + 1j * float(p[1])

    return psi


def check_selected_wavefunctions(
    psi: np.ndarray,
    k_label: int,
    norm_tol: float,
    orth_tol: float,
) -> tuple[float, float]:
    """
    Check wavefunction normalization and orthogonality.

    Parameters
    ----------
    psi
        Shape (nions,nbands).
    k_label
        k-point index for warnings.
    norm_tol
        Norm warning tolerance.
    orth_tol
        Orthogonality warning tolerance.

    Returns
    -------
    tuple
        (max_norm_error,max_offdiagonal_overlap).
    """
    overlap = psi.conj().T @ psi

    norm_error = float(
        np.max(np.abs(np.real(np.diag(overlap)) - 1.0))
    )

    offdiag = overlap - np.diag(np.diag(overlap))
    orth_error = float(np.max(np.abs(offdiag)))

    if norm_error > norm_tol:
        print(
            f"[warn] k={k_label}: wavefunction norm error={norm_error:.3e}"
        )

    if orth_error > orth_tol:
        print(
            f"[warn] k={k_label}: off-diagonal overlap={orth_error:.3e}"
        )

    return norm_error, orth_error


def calculate_g_matrix_at_k(
    plus: HoppingData,
    minus: HoppingData,
    k_frac: np.ndarray,
    psi0: np.ndarray,
    scale_factor_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate projected H_plus, H_minus and EPC G(k) at one k point.

    Parameters
    ----------
    plus, minus
        Plus/minus hopping data.
    k_frac
        Electronic k point.
    psi0
        Equilibrium eigenvectors.
    scale_factor_s
        Frozen-displacement scale s.

    Returns
    -------
    tuple
        (g,hplus_subspace,hminus_subspace,hermiticity_error).
    """
    nions = psi0.shape[0]

    hplus = build_hk_sparse(plus, k_frac, nions)
    hminus = build_hk_sparse(minus, k_frac, nions)

    hplus_psi = hplus @ psi0
    hminus_psi = hminus @ psi0

    hplus_sub = psi0.conj().T @ hplus_psi
    hminus_sub = psi0.conj().T @ hminus_psi

    g = (hplus_sub - hminus_sub) / (2.0 * scale_factor_s)

    denom = np.linalg.norm(g)
    herm_error = (
        0.0
        if denom == 0.0
        else float(np.linalg.norm(g - g.conj().T) / denom)
    )

    return g, hplus_sub, hminus_sub, herm_error


def write_long_g2_table(
    path: str | Path,
    kpoints: np.ndarray,
    selected_bands: list[int],
    g: np.ndarray,
) -> None:
    """
    Write one row for every (k,m,n).

    Parameters
    ----------
    path
        Output filename.
    kpoints
        Fractional reciprocal k points.
    selected_bands
        Global band labels.
    g
        EPC matrices.

    Returns
    -------
    None
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# ik0 ik1 k1 k2 k3 m_local n_local m_global n_global "
            "Re_g_eV Im_g_eV abs_g_eV abs_g2_eV2\n"
        )

        nk, nb, _ = g.shape

        for ik in range(nk):
            k1, k2, k3 = kpoints[ik]

            for m in range(nb):
                for n in range(nb):
                    z = g[ik, m, n]

                    f.write(
                        f"{ik:6d} {ik+1:6d} "
                        f"{k1: .12f} {k2: .12f} {k3: .12f} "
                        f"{m:4d} {n:4d} "
                        f"{selected_bands[m]:8d} {selected_bands[n]:8d} "
                        f"{z.real: .16e} {z.imag: .16e} "
                        f"{abs(z): .16e} {abs(z)**2: .16e}\n"
                    )


def write_subspace_table(
    path: str | Path,
    kpoints: np.ndarray,
    g: np.ndarray,
    herm_errors: np.ndarray,
    norm_errors: np.ndarray,
    orth_errors: np.ndarray,
) -> None:
    """
    Write gauge-robust subspace EPC summaries.

    Parameters
    ----------
    path
        Output filename.
    kpoints
        Fractional reciprocal k points.
    g
        EPC matrices.
    herm_errors, norm_errors, orth_errors
        Diagnostic arrays.

    Returns
    -------
    None
    """
    g2 = np.abs(g) ** 2
    total = np.sum(g2, axis=(1, 2))

    diag = np.diagonal(g, axis1=1, axis2=2)
    diag2 = np.sum(np.abs(diag) ** 2, axis=1)
    off2 = total - diag2
    max2 = np.max(g2, axis=(1, 2))

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# ik0 ik1 k1 k2 k3 sum_mn_abs_g2_eV2 "
            "diagonal_abs_g2_eV2 offdiagonal_abs_g2_eV2 "
            "max_abs_g2_eV2 hermiticity_error wf_norm_error wf_orth_error\n"
        )

        for ik, k in enumerate(kpoints):
            f.write(
                f"{ik:6d} {ik+1:6d} "
                f"{k[0]: .12f} {k[1]: .12f} {k[2]: .12f} "
                f"{total[ik]: .16e} {diag2[ik]: .16e} "
                f"{off2[ik]: .16e} {max2[ik]: .16e} "
                f"{herm_errors[ik]: .8e} {norm_errors[ik]: .8e} "
                f"{orth_errors[ik]: .8e}\n"
            )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "Gamma frozen-phonon EPC with independent Fourier transforms "
            "of H_plus and H_minus."
        )
    )

    p.add_argument("--hr-plus", required=True)
    p.add_argument("--hr-minus", required=True)
    p.add_argument("--wavefunction", required=True)
    p.add_argument("--kpoints", default="KPOINTS")
    p.add_argument("--metadata", default=None)
    p.add_argument("--scale-factor", type=float, default=None)

    p.add_argument(
        "--bands",
        nargs="+",
        type=int,
        required=True,
        help="Global band indices, e.g. --bands 5575 5576 ... 5580",
    )

    p.add_argument(
        "--wavefunction-k-start",
        type=int,
        default=1,
        help="Header k index corresponding to the first KPOINTS entry.",
    )

    p.add_argument("--norm-tol", type=float, default=1e-6)
    p.add_argument("--orth-tol", type=float, default=1e-6)
    p.add_argument("--herm-tol", type=float, default=1e-6)
    p.add_argument("--graph-samples", type=int, default=20)
    p.add_argument("--prefix", default="epc_gamma")

    return p.parse_args()


def main() -> None:
    """Run the Gamma-point frozen-phonon EPC calculation."""
    args = parse_args()

    if args.scale_factor is not None:
        s = float(args.scale_factor)
    elif args.metadata is not None:
        s = read_scale_factor_from_metadata(args.metadata)
    else:
        raise ValueError("Provide --metadata or --scale-factor.")

    if s <= 0:
        raise ValueError("scale factor must be positive.")

    bands = [int(x) for x in args.bands]

    if len(set(bands)) != len(bands):
        raise ValueError("--bands contains duplicates.")

    print("Gamma frozen-phonon EPC")
    print("=" * 68)
    print(f"  scale factor s : {s:.12g}")
    print(f"  selected bands : {bands}")

    print("\nReading H_plus and H_minus...")
    hp = read_hopping_list(args.hr_plus)
    hm = read_hopping_list(args.hr_minus)

    graph_report = f"{args.prefix}_graph_report.txt"
    graph_stats = check_hopping_graph_by_pair(
        hp,
        hm,
        graph_report,
        args.graph_samples,
    )

    kpoints = read_kpoints(args.kpoints)
    nk = len(kpoints)
    nb = len(bands)

    print("\nIndexing equilibrium wavefunction file...")
    wf_index = index_wavefunction_file(
        args.wavefunction,
        bands,
    )

    print(f"  wavefunction nions : {wf_index.nions}")
    print(f"  KPOINTS nkpts      : {nk}")

    if wf_index.nkpts_header is not None:
        print(f"  wavefunction nkpts : {wf_index.nkpts_header}")

    if hp.nions_inferred > wf_index.nions:
        raise ValueError("H_plus basis exceeds wavefunction basis size.")
    if hm.nions_inferred > wf_index.nions:
        raise ValueError("H_minus basis exceeds wavefunction basis size.")

    g_all = np.zeros((nk, nb, nb), dtype=np.complex128)
    hp_sub_all = np.zeros_like(g_all)
    hm_sub_all = np.zeros_like(g_all)

    herm = np.zeros(nk)
    normerr = np.zeros(nk)
    ortherr = np.zeros(nk)

    print("\nCalculating EPC matrices...")

    for ik0, k in enumerate(kpoints):
        ik_file = args.wavefunction_k_start + ik0

        psi0 = read_selected_wavefunctions_at_k(
            args.wavefunction,
            wf_index,
            ik_file,
            bands,
        )

        normerr[ik0], ortherr[ik0] = check_selected_wavefunctions(
            psi0,
            ik_file,
            args.norm_tol,
            args.orth_tol,
        )

        (
            g_all[ik0],
            hp_sub_all[ik0],
            hm_sub_all[ik0],
            herm[ik0],
        ) = calculate_g_matrix_at_k(
            hp,
            hm,
            k,
            psi0,
            s,
        )

        if herm[ik0] > args.herm_tol:
            print(
                f"[warn] k={ik_file}: relative Hermiticity error "
                f"of G = {herm[ik0]:.3e}"
            )

        if ik0 == 0 or (ik0 + 1) % 10 == 0 or ik0 + 1 == nk:
            print(f"  processed {ik0+1:5d}/{nk}")

    g2 = np.abs(g_all) ** 2
    subspace_g2 = np.sum(g2, axis=(1, 2))

    npz_path = f"{args.prefix}_results.npz"

    np.savez_compressed(
        npz_path,
        kpoints_frac=kpoints,
        selected_bands=np.asarray(bands, dtype=np.int64),
        scale_factor_s=np.asarray(s),
        g_complex_eV=g_all,
        g2_eV2=g2,
        hplus_subspace_eV=hp_sub_all,
        hminus_subspace_eV=hm_sub_all,
        subspace_sum_g2_eV2=subspace_g2,
        hermiticity_error=herm,
        wavefunction_norm_error=normerr,
        wavefunction_orthogonality_error=ortherr,
        graph_only_plus_pairs=np.asarray(
            graph_stats["only_plus_pairs"], dtype=np.int64
        ),
        graph_only_minus_pairs=np.asarray(
            graph_stats["only_minus_pairs"], dtype=np.int64
        ),
    )

    long_path = f"{args.prefix}_g2_long.dat"
    sub_path = f"{args.prefix}_subspace.dat"

    write_long_g2_table(long_path, kpoints, bands, g_all)
    write_subspace_table(
        sub_path,
        kpoints,
        g_all,
        herm,
        normerr,
        ortherr,
    )

    print("\nFinished")
    print("=" * 68)
    print(f"  results npz    : {npz_path}")
    print(f"  long g2 table  : {long_path}")
    print(f"  subspace table : {sub_path}")
    print(f"  graph report   : {graph_report}")
    print(f"  max |g|        : {np.max(np.abs(g_all)):.8e} eV")
    print(f"  max |g|^2      : {np.max(g2):.8e} eV^2")
    print(f"  max Herm error : {np.max(herm):.8e}")
    print(f"  max norm error : {np.max(normerr):.8e}")
    print(f"  max orth error : {np.max(ortherr):.8e}")


if __name__ == "__main__":
    main()
