from collections import defaultdict
import numpy as np

from ase.io import read

def read_hopping_list(path):
    """
    Read a real-space TB hopping file.

    Expected columns
    ----------------
    i j R1_frac R2_frac R3_frac Re Im

    Returns
    -------
    list
        Each entry is (i, j, r_frac, hopping).
    """
    data = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 7:
                continue

            i = int(parts[0])
            j = int(parts[1])

            r = np.array(
                [
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ],
                dtype=float,
            )

            val = (
                float(parts[5])
                + 1j * float(parts[6])
            )

            data.append(
                (i, j, r, val)
            )

    return data


def group_by_ij(hops):
    """
    Group hopping entries by orbital pair (i,j).

    Parameters
    ----------
    hops
        Hopping list.

    Returns
    -------
    dict
        (i,j) -> list[(R, hopping)].
    """
    groups = defaultdict(list)

    for i, j, r, val in hops:
        groups[(i, j)].append(
            (r, val)
        )

    return groups


def compare_hopping_graph(
    plus_path,
    minus_path,
):
    """
    Compare H_plus and H_minus hopping topology without requiring R to be equal.

    This distinguishes:
      1. genuine disappearance/appearance of an (i,j) hopping;
      2. the same hopping whose R vector merely changed under displacement.
    """
    hp = read_hopping_list(
        plus_path
    )

    hm = read_hopping_list(
        minus_path
    )

    gp = group_by_ij(
        hp
    )

    gm = group_by_ij(
        hm
    )

    kp = set(
        gp.keys()
    )

    km = set(
        gm.keys()
    )

    print("=" * 70)
    print("Basic statistics")
    print("=" * 70)

    print(
        f"H_plus entries  : {len(hp)}"
    )
    print(
        f"H_minus entries : {len(hm)}"
    )

    print(
        f"unique (i,j) plus  : {len(kp)}"
    )
    print(
        f"unique (i,j) minus : {len(km)}"
    )

    print()
    print(
        f"(i,j) only in plus  : {len(kp-km)}"
    )
    print(
        f"(i,j) only in minus : {len(km-kp)}"
    )

    common_pairs = (
        kp & km
    )

    count_different = []

    for key in common_pairs:
        np_ = len(
            gp[key]
        )
        nm_ = len(
            gm[key]
        )

        if np_ != nm_:
            count_different.append(
                (
                    key,
                    np_,
                    nm_,
                )
            )

    print()
    print(
        "Common (i,j) pairs with different "
        f"number of hopping terms: {len(count_different)}"
    )

    if count_different:
        print()
        print(
            "First 20 examples:"
        )

        for item in count_different[:20]:
            print(
                item
            )
    only_plus = kp - km
    only_minus = km - kp

    plus_values = []
    minus_values = []
    plus_r=[]
    minus_r=[]
    for key in only_plus:
        for r, val in gp[key]:
            plus_values.append(abs(val))
            plus_r.append(r)

    for key in only_minus:
        for r, val in gm[key]:
            minus_values.append(abs(val))
            minus_r.append(r)
    plus_values = np.array(plus_values)
    minus_values = np.array(minus_values)
    plus_r = np.array(plus_r)
    minus_r = np.array(minus_r)
    plus_r_data = calculate_distance(plus_r, "./plus/POSCAR")
    minus_r_data = calculate_distance(minus_r, "./minus/POSCAR")
    np.savetxt("plus_r.dat", plus_r_data)
    np.savetxt("minus_r.dat", minus_r_data)
    print("\nExclusive hopping amplitudes")
    print("=" * 70)

    print("Only in H_plus:")
    print("  number =", len(plus_values))
    print("  min    =", np.min(plus_values))
    print("  mean   =", np.mean(plus_values))
    print("  median =", np.median(plus_values))
    print("  max    =", np.max(plus_values))

    print("\nOnly in H_minus:")
    print("  number =", len(minus_values))
    print("  min    =", np.min(minus_values))
    print("  mean   =", np.mean(minus_values))
    print("  median =", np.median(minus_values))
    print("  max    =", np.max(minus_values))


def calculate_distance(d_fracs, structure_file):
    atoms = read(structure_file)
    cell = atoms.cell.array
    distances = np.linalg.norm(d_fracs @ cell, axis=1)
    r_cart=d_fracs @ cell
    result=np.concatenate([r_cart, distances[:, np.newaxis]], axis=1)
    return result

if __name__ == "__main__":

    compare_hopping_graph(
        plus_path="plus/tb_hr.dat",
        minus_path="minus/tb_hr.dat",
    )
