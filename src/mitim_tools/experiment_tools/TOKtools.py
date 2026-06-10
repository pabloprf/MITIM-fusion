
# Generic, simplified ICRF antenna parameters for use in MITIM when machine_structures is null.
# These are not meant to be realistic, but rather to provide a simple setup for testing and development purposes.

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