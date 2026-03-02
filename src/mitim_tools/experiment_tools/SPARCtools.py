from mitim_tools.misc_tools import IOtools
from IPython import embed

# Location of the FW description (non open source) -------------------
LimiterPath = "$MFEIM_PATH/private_code_mitim/FREEGS_SPARC/fw/sparc_FW.txt"
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
        "nicha     = 2         ! Number of ICRH antennae",
        f"frqicha(1)   = {MHz}e6   ! Frequency of antenna 1 (Hz)",
        f"frqicha(2)   = {MHz}e6   ! Frequency of antenna 2 (Hz)",
        "rgeoant_a(1,1)   = 242.05",
        "rgeoant_a(2,1)   = 245.0",
        "ygeoant_a(1,1)   = 37.055",
        "ygeoant_a(2,1)   = 4.555",
        "rgeoant_a(1,2)   = 242.05",
        "rgeoant_a(2,2)   = 245.0",
        "ygeoant_a(1,2)   = -37.62",
        "ygeoant_a(2,2)   = -5.12",
        "num_nphi(1)      = 1      ! Num of Nphi per antenna",
        "nnphi(1,1)       = 38     ! Nphi values",
        "wnphi(1,1)       = 1.     ! Nphi power weightings",
        "num_nphi(2)      = 1      ! Num of Nphi per antenna",
        "nnphi(1,2)       = 38     ! Nphi values",
        "wnphi(1,2)       = 1.     ! Nphi power weightings",
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
