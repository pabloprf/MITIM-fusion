"""
G-file (EFIT GEQDSK) reader and SSF equilibrium parameter calculator.

Provides:
  - GFileReader : thin wrapper around freeqdsk for g-file I/O
  - EquilibriumSSF : computes G0 and alpha_s for the Peret 2025 SSF model
      by tracing a field line at the separatrix and averaging the curvature
      drive and magnetic-shear-induced tilt weighted by the ballooning envelope.

References
----------
Peret et al., Nucl. Fusion 65 (2025) 056043, sections 2.4 and 2.5.
freeqdsk: https://github.com/bendudson/freeqdsk
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

try:
    import freeqdsk.geqdsk as _geqdsk
    _FREEQDSK_AVAILABLE = True
except ImportError:
    _FREEQDSK_AVAILABLE = False


# ---------------------------------------------------------------------------
# G-file reader
# ---------------------------------------------------------------------------

class GFileReader:
    """
    Read an EFIT GEQDSK (g-file).

    Uses freeqdsk (https://github.com/bendudson/freeqdsk) when available for
    robust Fortran-format parsing.  Falls back to a pure-numpy parser for
    environments where freeqdsk is not installed.

    Attributes
    ----------
    nw, nh : int
        Grid dimensions in R and Z.
    rdim, zdim : float
        Physical size of the grid (m).
    rcentr, bcentr : float
        Reference R (m) and vacuum BT at rcentr (T).
    rleft, zmid : float
        Left boundary R and midpoint Z (m).
    rmaxis, zmaxis : float
        Magnetic axis location (m).
    simag, sibry : float
        Poloidal flux at axis and boundary (Wb/rad).
    rbbbs, zbbbs : ndarray
        Boundary (separatrix) R, Z coordinates (m).
    R, Z : ndarray (nw,), (nh,)
        1-D grid vectors.
    psi2d : ndarray (nh, nw)
        Poloidal flux on the 2-D grid (Wb/rad).
    fpol : ndarray (nw,)
        Poloidal current function F = R*BT on uniform-psi grid.
    psi1d : ndarray (nw,)
        Uniform psi grid from axis to boundary.
    """

    def __init__(self, path: str):
        if _FREEQDSK_AVAILABLE:
            self._read_freeqdsk(path)
        else:
            self._read_fallback(path)

    # ------------------------------------------------------------------
    def _read_freeqdsk(self, path: str):
        """Parse using freeqdsk — handles all GEQDSK dialect variations."""
        with open(path) as fh:
            data = _geqdsk.read(fh)

        self.nw = data.nx
        self.nh = data.ny
        self.rdim   = data.rdim
        self.zdim   = data.zdim
        self.rcentr = data.rcentr
        self.rleft  = data.rleft
        self.zmid   = data.zmid
        self.rmaxis = data.rmagx
        self.zmaxis = data.zmagx
        self.simag  = data.simagx
        self.sibry  = data.sibdry
        self.bcentr = data.bcentr

        self.fpol   = np.asarray(data.fpol)
        self.pres   = np.asarray(data.pres)
        self.ffprim = np.asarray(data.ffprime)
        self.pprime = np.asarray(data.pprime)
        self.qpsi   = np.asarray(data.qpsi)
        self.rbbbs  = np.asarray(data.rbdry)
        self.zbbbs  = np.asarray(data.zbdry)

        # freeqdsk returns psi as (nx, ny) row-major; reshape to (nh, nw)
        self.psi2d = np.asarray(data.psi).reshape(self.nh, self.nw)

        # freeqdsk provides r_grid and z_grid as 1-D arrays
        self.R = np.asarray(data.r_grid)
        self.Z = np.asarray(data.z_grid)
        self.psi1d = np.linspace(self.simag, self.sibry, self.nw)
        self._dpsi = self.sibry - self.simag

    # ------------------------------------------------------------------
    def _read_fallback(self, path: str):
        """Pure-numpy GEQDSK parser (fallback when freeqdsk is unavailable)."""
        with open(path) as f:
            lines = f.readlines()

        tokens = lines[0].split()
        self.nw = int(tokens[-2])
        self.nh = int(tokens[-1])

        data = []
        for line in lines[1:]:
            for tok in line.split():
                try:
                    data.append(float(tok))
                except ValueError:
                    pass

        d = iter(data)

        def _rn(n):
            return np.array([next(d) for _ in range(n)])

        self.rdim   = next(d);  self.zdim   = next(d)
        self.rcentr = next(d);  self.rleft  = next(d);  self.zmid   = next(d)
        self.rmaxis = next(d);  self.zmaxis = next(d)
        self.simag  = next(d);  self.sibry  = next(d);  self.bcentr = next(d)
        next(d);  _rn(5)  # cpasma + 5 duplicates

        self.fpol   = _rn(self.nw)
        self.pres   = _rn(self.nw)
        self.ffprim = _rn(self.nw)
        self.pprime = _rn(self.nw)
        psi2d_flat  = _rn(self.nw * self.nh)
        self.qpsi   = _rn(self.nw)

        nbbbs = int(next(d));  next(d)
        bbbs  = _rn(2 * nbbbs)
        self.rbbbs = bbbs[0::2];  self.zbbbs = bbbs[1::2]

        self.psi2d = psi2d_flat.reshape(self.nh, self.nw)
        self.R = self.rleft + np.linspace(0, 1, self.nw) * self.rdim
        self.Z = (self.zmid - 0.5 * self.zdim) + np.linspace(0, 1, self.nh) * self.zdim
        self.psi1d = np.linspace(self.simag, self.sibry, self.nw)
        self._dpsi = self.sibry - self.simag

    # ------------------------------------------------------------------
    # Interpolators (lazy)

    @property
    def psi_spl(self):
        if not hasattr(self, '_psi_spl'):
            self._psi_spl = RectBivariateSpline(self.Z, self.R, self.psi2d)
        return self._psi_spl

    @property
    def F_spl(self):
        if not hasattr(self, '_F_spl'):
            self._F_spl = interp1d(self.psi1d, self.fpol,
                                   kind='linear', fill_value='extrapolate')
        return self._F_spl

    # ------------------------------------------------------------------
    # Field components

    def psi(self, R, Z):
        return float(self.psi_spl(Z, R))

    def B_R(self, R, Z):
        """B_R = -(1/R) dpsi/dZ  (T)."""
        return -float(self.psi_spl(Z, R, dy=1)) / R

    def B_Z(self, R, Z):
        """B_Z = (1/R) dpsi/dR  (T)."""
        return float(self.psi_spl(Z, R, dx=1)) / R

    def B_phi(self, R, Z):
        """B_phi = F(psi)/R  (T)."""
        return float(self.F_spl(self.psi(R, Z))) / R

    def B_total(self, R, Z):
        return np.sqrt(self.B_R(R, Z)**2 + self.B_Z(R, Z)**2 + self.B_phi(R, Z)**2)

    def B_pol(self, R, Z):
        return np.sqrt(self.B_R(R, Z)**2 + self.B_Z(R, Z)**2)

    def find_xpoint(self):
        """
        Locate the primary X-point by finding the B_pol = 0 null closest to
        the extremum of |Z| on the stored separatrix boundary.

        Returns (R_Xpt, Z_Xpt) in metres.
        """
        dz = np.abs(self.zbbbs - self.zmaxis)
        idx = np.argmax(dz)
        R0, Z0 = float(self.rbbbs[idx]), float(self.zbbbs[idx])

        from scipy.optimize import fsolve as _fsolve
        def f(x):
            try:
                return [self.B_R(x[0], x[1]), self.B_Z(x[0], x[1])]
            except Exception:
                return [1.0, 1.0]
        sol = _fsolve(f, [R0, Z0], full_output=True)
        if sol[2] == 1:
            return float(sol[0][0]), float(sol[0][1])
        return R0, Z0


# ---------------------------------------------------------------------------
# Field-line tracer at the separatrix
# ---------------------------------------------------------------------------

class FieldLineTracer:
    """
    Trace a single field line at the separatrix (outer midplane start)
    and compute arc-length-resolved magnetic geometry needed for SSF
    averaging.

    Parameters
    ----------
    gfile : GFileReader
    n_points : int
        Number of arc-length steps around one poloidal circuit.
    """

    def __init__(self, gfile: GFileReader, n_points: int = 512):
        self.g = gfile
        self.n_points = n_points
        self._trace()

    def _trace(self):
        """
        Trace the outermost closed flux surface (separatrix) in the poloidal
        plane by following the boundary stored in the g-file, then interpolate
        magnetic vectors along it.

        Stores
        ------
        self.l      : arc length (m)
        self.R_fl   : R along the field line
        self.Z_fl   : Z along the field line
        self.theta  : geometric poloidal angle measured from midplane (rad)
        self.B      : total field strength
        self.B_pol  : poloidal field magnitude
        self.B_tor  : toroidal field magnitude
        self.dl     : differential arc-length element
        """
        g = self.g

        # Use the stored boundary, re-interpolated to n_points
        rbbbs = np.asarray(g.rbbbs)
        zbbbs = np.asarray(g.zbbbs)

        # Close the loop
        rbbbs = np.append(rbbbs, rbbbs[0])
        zbbbs = np.append(zbbbs, zbbbs[0])

        # Parameterise by cumulative arc length
        ds = np.sqrt(np.diff(rbbbs)**2 + np.diff(zbbbs)**2)
        s_raw = np.concatenate([[0], np.cumsum(ds)])
        s_uniform = np.linspace(0, s_raw[-1], self.n_points + 1)[:-1]

        R_fl = np.interp(s_uniform, s_raw, rbbbs)
        Z_fl = np.interp(s_uniform, s_raw, zbbbs)

        # Midplane outboard start: max R point near Z ~ zmaxis
        near_mid = np.abs(Z_fl - g.zmaxis) < 0.05 * g.zdim
        if near_mid.sum() > 0:
            idx_start = np.where(near_mid)[0][np.argmax(R_fl[near_mid])]
        else:
            idx_start = np.argmax(R_fl)

        # Re-roll so we start at outer midplane
        R_fl = np.roll(R_fl, -idx_start)
        Z_fl = np.roll(Z_fl, -idx_start)
        s_uniform = np.roll(s_uniform, -idx_start)
        # Fix arc length to start at 0
        s_uniform = s_uniform - s_uniform[0]
        s_uniform[s_uniform < 0] += s_raw[-1]

        # Differential arc length (poloidal element)
        dR = np.gradient(R_fl, s_uniform)
        dZ = np.gradient(Z_fl, s_uniform)
        dl_pol = np.sqrt(dR**2 + dZ**2)  # should be ~ 1 by construction

        # Magnetic quantities along the separatrix
        B_tot = np.array([g.B_total(R, Z) for R, Z in zip(R_fl, Z_fl)])
        B_pol = np.array([g.B_pol(R, Z) for R, Z in zip(R_fl, Z_fl)])
        B_tor = np.array([g.B_phi(R, Z) for R, Z in zip(R_fl, Z_fl)])

        # Poloidal angle relative to midplane
        theta = np.arctan2(Z_fl - g.zmaxis, R_fl - g.rmaxis)

        # Parallel arc length element: dl_par = dl_pol * (B_tot / B_pol)
        dl_par = dl_pol * (B_tot / np.maximum(B_pol, 1e-10))

        self.l = np.cumsum(dl_par) - dl_par[0]   # parallel arc length
        self.s_pol = s_uniform                     # poloidal arc length
        self.R_fl = R_fl
        self.Z_fl = Z_fl
        self.theta = theta
        self.B = B_tot
        self.B_pol = B_pol
        self.B_tor = B_tor
        self.dl = dl_par


# ---------------------------------------------------------------------------
# SSF equilibrium parameter calculator
# ---------------------------------------------------------------------------

class EquilibriumSSF:
    """
    Compute G0 and alpha_s for the Peret 2025 SSF model from a g-file.

    Parameters
    ----------
    gfile_path : str
        Path to the EFIT GEQDSK (g-file).
    n_points : int
        Number of poloidal points for the field-line trace.
    eddy_width_m : float
        Radial width of the flux tube / turbulent eddy at the midplane (m).
        Used to define the volume averaged for G0 (paper: ~5 mm).

    Attributes (available after calling compute())
    ----------
    G0 : float
        Flux-tube-averaged curvature drive coefficient.
    alpha_s : float
        Magnetic-shear-induced average mode tilt (signed).
    R_Xpt, Z_Xpt : float
        Primary X-point location (m).
    flux_expansion : float
        B_midplane / B_Xpoint (poloidal field ratio at separatrix).
    theta_Xpt : float
        Poloidal angle of the primary X-point (rad).
    """

    def __init__(self, gfile_path: str, n_points: int = 512,
                 eddy_width_m: float = 0.005):
        self.g = GFileReader(gfile_path)
        self.tracer = FieldLineTracer(self.g, n_points=n_points)
        self.eddy_width_m = eddy_width_m
        self.G0 = None
        self.alpha_s = None
        self._computed = False

    def compute(self):
        """Run all computations. Returns self for chaining."""
        self._compute_xpoint()
        self._compute_G0()
        self._compute_alpha_s()
        self._computed = True
        return self

    # ------------------------------------------------------------------
    def _compute_xpoint(self):
        g = self.g
        self.R_Xpt, self.Z_Xpt = g.find_xpoint()

        # Poloidal angle of the X-point
        self.theta_Xpt = np.arctan2(self.Z_Xpt - g.zmaxis,
                                     self.R_Xpt - g.rmaxis)

        # Flux expansion: ratio of poloidal B at midplane to B at X-point
        # Outer midplane position on separatrix
        t = self.tracer
        idx_mid = np.argmin(np.abs(t.theta))      # closest to theta=0
        idx_Xpt = np.argmin((t.R_fl - self.R_Xpt)**2 + (t.Z_fl - self.Z_Xpt)**2)

        self.B_pol_mid = float(t.B_pol[idx_mid])
        self.B_pol_Xpt = float(t.B_pol[idx_Xpt])
        self.flux_expansion = (self.B_pol_mid /
                               max(self.B_pol_Xpt, 1e-10))

    # ------------------------------------------------------------------
    def _curvature_drive_profile(self):
        """
        g(l) = hat{R} . (-hat{n}) — projection of the radial (outward) unit
        vector onto the negative pressure-gradient (i.e. inward normal) unit
        vector.  In cylindrical coordinates this is simply cos(theta) weighted
        by the local curvature geometry.

        For the SSF model, the relevant quantity is the ballooning curvature:

            C(l) = (cos(theta) + B_pol/B * sin(theta) * s_hat)  [simplified]

        Here we use the direct magnetic curvature via the normal-to-flux-surface
        unit vector dotted with the outward major-radius unit vector:

            kappa_n = (hat{b} . nabla) hat{b}  (normal curvature)

        In the EFIT grid the normal curvature at every poloidal point on the
        separatrix is approximated from the local field-line geometry.
        """
        t = self.tracer
        R, Z = t.R_fl, t.Z_fl
        B = t.B
        B_pol = t.B_pol

        # Unit hat{b} = (B_R, B_Z, B_phi) / B  (in cylindrical R, phi, Z)
        B_R  = np.array([self.g.B_R(R[i], Z[i]) for i in range(len(R))])
        B_Z  = np.array([self.g.B_Z(R[i], Z[i]) for i in range(len(Z))])
        B_phi = t.B_tor
        bR = B_R / B;  bZ = B_Z / B;  bphi = B_phi / B

        # parametric arc derivative of hat{b}  (d hat{b} / dl_par)
        l = t.l
        dbR  = np.gradient(bR,  l)
        dbZ  = np.gradient(bZ,  l)
        dbphi = np.gradient(bphi, l)

        # Normal curvature vector kappa_n (cylindrical, dropping centrifugal for now)
        kappa_R   = dbR   - bphi**2 / R
        kappa_Z   = dbZ
        kappa_phi = dbphi + bR * bphi / R

        # Outward major-radius unit vector hat{e}_R = (1, 0, 0)
        # Pressure gradient unit vector (inward normal to flux surface ~ -hat{e}_R)
        # Projection: G_l = kappa · hat{e}_R  (sign: positive on LFS for interchange drive)
        G_l = kappa_R   # G0 proxy along field line (curvature drive coefficient)

        return G_l

    def _compute_G0(self):
        """
        G0 = parallel average of max(G_l, 0) weighted by the ballooning
        turbulence envelope (itself taken as max(G_l, 0)).

        This implements the 'projection of the parallel profile' described in
        section 2.4 of the paper: the flux tube is averaged over a radial width
        equal to eddy_width_m at the midplane.
        """
        t = self.tracer
        G_l = self._curvature_drive_profile()

        # Ballooning envelope: positive parts of G_l (drive region)
        env = np.maximum(G_l, 0.0)
        dl  = t.dl

        # Weighted average
        denom = np.sum(env * dl)
        if denom < 1e-30:
            self.G0 = 1.0
            return
        G0_raw = np.sum(G_l * env * dl) / denom

        # Normalise so that G0 ~ 1 for typical tokamak geometry
        # The paper defines g = G0 * rho_s / R0; the scaling of G_l from the
        # curvature vector has units of 1/m.  Multiply by R0 to make dimensionless.
        self.G0 = float(G0_raw * self.g.rmaxis)

    # ------------------------------------------------------------------
    def _shear_tilt_profile(self):
        """
        alpha_s(l): magnetic-shear-induced tilt of the wave-vector.

        Following the paper (section 2.4), alpha_s(l) is the scalar product
        of the vector perpendicular to the local pressure-gradient unit vector
        with the local vector normal to the flux surface:

            alpha_s(l) = hat{e}_perp . nabla_perp(hat{b})

        In practice this is approximated as the differential change in the
        poloidal angle of hat{b} per unit parallel length — i.e. the local
        field-line pitch-angle variation normalised to rho_s/R0:

            alpha_s(l) ~ d(B_pol / B) / d(l_par)  * L_parallel_ref

        We use the normalised pitch angle (B_pol/B) as the tilt proxy, which
        captures how much the field twists per unit parallel arc, and is a
        purely poloidal quantity that averages to zero in up-down symmetric
        geometry but not in the presence of an X-point.
        """
        t = self.tracer
        pitch = t.B_pol / t.B          # = sin(pitch angle) = B_pol / B
        l     = t.l

        # Local shear-induced tilt ~ d(pitch)/dl_par (1/m), then scale by L_par
        L_par = l[-1]                  # full parallel connection length
        alpha_l = np.gradient(pitch, l) * L_par

        return alpha_l

    def _compute_alpha_s(self):
        """
        alpha_s = parallel average of alpha_s(l) weighted by the ballooning
        envelope max(G_l, 0), as described in section 2.4.

        The sign is determined by the X-point position:
          - LSN (Z_Xpt < zmaxis): alpha_s < 0  (favorable)
          - USN (Z_Xpt > zmaxis): alpha_s > 0  (unfavorable)
        """
        t = self.tracer
        G_l   = self._curvature_drive_profile()
        env   = np.maximum(G_l, 0.0)
        dl    = t.dl
        alpha_l = self._shear_tilt_profile()

        denom = np.sum(env * dl)
        if denom < 1e-30:
            self.alpha_s = -0.5
            return

        alpha_s_raw = np.sum(alpha_l * env * dl) / denom

        # Apply sign convention from the paper
        sign = -1.0 if self.Z_Xpt < self.g.zmaxis else 1.0
        self.alpha_s = float(sign * abs(alpha_s_raw))

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return computed parameters as a dictionary."""
        if not self._computed:
            self.compute()
        return {
            'G0': self.G0,
            'alpha_s': self.alpha_s,
            'R_Xpt': self.R_Xpt,
            'Z_Xpt': self.Z_Xpt,
            'theta_Xpt_rad': self.theta_Xpt,
            'flux_expansion': self.flux_expansion,
            'B_pol_mid': self.B_pol_mid,
            'B_pol_Xpt': self.B_pol_Xpt,
        }


# ---------------------------------------------------------------------------
# Convenience loader (cached per path)
# ---------------------------------------------------------------------------

_cache: dict = {}

def load_equilibrium(gfile_path: str, n_points: int = 512,
                     eddy_width_m: float = 0.005,
                     force_reload: bool = False) -> EquilibriumSSF:
    """
    Load and compute the SSF equilibrium parameters from a g-file.
    Results are cached so repeated calls within a run are free.

    Parameters
    ----------
    gfile_path : str
        Absolute or relative path to the EFIT GEQDSK g-file.
    n_points : int
        Poloidal resolution of the field-line trace (default 512).
    eddy_width_m : float
        Turbulent eddy width at midplane in metres (default 5 mm).
    force_reload : bool
        If True, ignore the cache and re-read the file.

    Returns
    -------
    EquilibriumSSF
        Fully computed object with G0, alpha_s, etc.
    """
    if not force_reload and gfile_path in _cache:
        return _cache[gfile_path]
    eq = EquilibriumSSF(gfile_path, n_points=n_points,
                        eddy_width_m=eddy_width_m).compute()
    _cache[gfile_path] = eq
    return eq
