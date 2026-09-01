import numpy as np
from ase.io import read


def inspect_hopping_distances(hr_file, structure_file):
    """
    Inspect the distance range of hopping entries.

    Parameters
    ----------
    hr_file
        TB hopping-list file.
    structure_file
        Structure file providing lattice vectors.

    Returns
    -------
    None
        Prints distance statistics.
    """
    atoms = read(structure_file)
    cell = atoms.cell.array

    distances = []

    with open(hr_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            p = line.split()

            if len(p) < 7:
                continue

            r_frac = np.array(
                [
                    float(p[2]),
                    float(p[3]),
                    float(p[4]),
                ]
            )

            r_cart = r_frac @ cell
            distances.append(
                np.linalg.norm(r_cart)
            )

    distances = np.array(distances)

    print("Number of hoppings :", len(distances))
    print("Min distance       :", distances.min(), "A")
    print("Median distance    :", np.median(distances), "A")
    print("Max distance       :", distances.max(), "A")

    for p in [90, 95, 99, 99.9, 99.99]:
        print(
            f"P{p:<5}             :",
            np.percentile(distances, p),
            "A",
        )


inspect_hopping_distances(
    "./equi/tb_hr.dat",
    "./equi/POSCAR",
)
