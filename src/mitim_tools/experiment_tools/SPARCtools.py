import os
from IPython import embed
from mitim_tools.misc_tools import IOtools

# Location of the FW description (non open source) -------------------
LimiterPath = "$STUDIES_PATH/information/FREEGS_SPARC/fw/sparc_FW.txt"
# --------------------------------------------------------------------


def defineTRANSPnmlStructures():
    limiters = [
        [103.50, 0.00, 90.00],
        [165.00, 142.63, 0.00],
        [
            236.49,
            0.00,
            90.00,
        ],
        [165.00, -142.63, 0.00],
    ]

    VVmoms = [
        [152.338524028, 0.000699170920186],
        [68.8351908266, 125.991190581],
        [12.5609291825, -0.000759580226652],
        [-0.00129837882638, -0.00747608916615],
        [-0.00239356869422, -0.000881576191633],
    ]

    return limiters, VVmoms


def defineFirstWall(file_rel=IOtools.expandPath(LimiterPath)):
    with open(f"{file_rel}", "r") as f:
        aux = f.readlines()

    r, z = [], []
    for line in aux:
        try:
            nums = [float(i) for i in line.split()]
        except:
            continue
        r.append(nums[0])
        z.append(nums[1])

    return r, z


def ICRFantennas(MHz=120.0):
    lines = [
        "! ----- Antenna Parameters",
        "nicha     = 1         ! Number of ICRH antennae",
        f"frqicha   = {MHz}e6   ! Frequency of antenna (Hz)",
        "!prficha    = 0.0      ! Power of antenna (W)",
        "rfartr    = 2.0       ! Distance (cm) from antenna for Faraday shield",
        "ngeoant   = 1         ! Geometry representation of antenna (1=traditional)",
        "rmjicha   = 165.0     ! Major radius of antenna (cm)",
        "rmnicha   = 55.0      ! Minor radius of antenna (cm)",
        "thicha    = 26.28     ! Theta extent of antenna (degrees)",
        "num_nphi  = 1         ! Num of Nphi per antenna",
        "nnphi     = 30        ! Nphi values",
        "wnphi     = 1.        ! Nphi power weightings",
        "",
    ]

    return "\n".join(lines)


def defineISOLVER():
    isolver_file = "file: iso_sprc.nc"

    pfcs = {
        "cs1": [-1, -0.039e6],
        "cs2u": [-1, -0.0016e6],
        "cs2l": [-1, -0.0016e6],
        "cs3u": [-1, 0.0145e6],
        "cs3l": [-1, 0.0145e6],
        "pf1u": [0, 0.0],
        "pf1l": [0, 0.0],
        "pf2u": [0, 0.0],
        "pf2l": [0, 0.0],
        "pf3u": [0, 0.0],
        "pf3l": [0, 0.0],
        "pf4u": [0, 0.0],
        "pf4l": [0, 0.0],
        "dv1u": [-1, 0.0],
        "dv1l": [-1, 0.0],
        "dv2u": [-1, 0.0],
        "dv2l": [-1, 0.0],
        "vs1": [-1, 0.0],
    }

    return isolver_file, pfcs
