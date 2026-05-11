"""
Parameterization models for plasma profiles (n_e, T_e, T_i, ...).

Provides a common interface via ParameterModel and concrete implementations:
- SplineParameterModel (Akima or PCHIP), parameterizing a/Ly at user-defined knots
- MTanhParameterModel (stub)
- GaussianRBFParameterModel (stub)

A factory function create_parameter_model(config) instantiates the requested model.

Conventions
-----------
- Coordinate x: parameterizer classes operate on normalized radius x = r/a ("roa").
    In this coordinate, a/Ly = - d(ln y)/d x, which integrates naturally.
- Boundary condition: to reconstruct y from gradients, a boundary value y_sep at the
  outermost grid point is required. Pass explicitly to .y(..., y_sep=...), or provide
  bc_field in options to read from state.BC.<bc_field> when state is supplied.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union, List
import numpy as np
import torch
import copy
from scipy.interpolate import InterpolatedUnivariateSpline as linear, Akima1DInterpolator as akima, PchipInterpolator as pchip, CubicSpline
from scipy.special import erf
from scipy.optimize import curve_fit
import scipy as sp  
from scipy.optimize import least_squares
from scipy.special import gamma as Gamma
from scipy.integrate import cumulative_trapezoid
from mitim_modules.powertorch.utils import CALCtools
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# -------------------------
# Legacy form for powerstate and portals_main
# -------------------------


def piecewise_linear(
    x_coord,
    y_coord_raw,
    x_coarse_tensor,
    parameterize_in_aLx=True,
    multiplier_quantity=1.0,
    ):
    """
    Notes:
        - x_coarse_tensor must be torch
    """

    # **********************************************************************************************************
    # Define the integrator and derivator functions (based on whether I want to parameterize in aLx or in gradX)
    # **********************************************************************************************************

    if parameterize_in_aLx:
        # 1/Lx = -1/X*dX/dr
        integrator_function, derivator_function = (
            CALCtools.integration_Lx,
            CALCtools.derivation_into_Lx,
        )
    else:
        # -dX/dr
        integrator_function, derivator_function = (
            CALCtools.integration_dxdr,
            CALCtools.derivation_into_dxdr,
        )

    y_coord = torch.from_numpy(y_coord_raw).to(x_coarse_tensor) * multiplier_quantity

    ygrad_coord = derivator_function( torch.from_numpy(x_coord).to(x_coarse_tensor), y_coord )

    # **********************************************************************************************************
    # Get control points
    # **********************************************************************************************************

    x_coarse = x_coarse_tensor[1:].cpu().numpy()

    """
    Define region to get control points from
    ------------------------------------------------------------
	Trick: Addition of extra point
		This is important because if I don't, when I combine the trailing edge and the new
		modified profile, there's going to be a discontinuity in the gradient.
	"""
    
    ir_end = np.argmin(np.abs(x_coord - x_coarse[-1]))

    if ir_end < len(x_coord) - 1:
        ir = ir_end + 2  # To prevent that TGYRO does a 2nd order derivative
        x_coarse = np.append(x_coarse, [x_coord[ir]])
    else:
        ir = ir_end

	# Definition of trailing edge. Any point after, and including, the extra point
    x_trail = torch.from_numpy(x_coord[ir:]).to(x_coarse_tensor)
    y_trail = y_coord[ir:]
    x_notrail = torch.from_numpy(x_coord[: ir + 1]).to(x_coarse_tensor)

    # Produce control points, including a zero at the beginning
    aLy_coarse = [[0.0, 0.0]]
    for cont, i in enumerate(x_coarse):
        yValue = ygrad_coord[np.argmin(np.abs(x_coord - i))]
        aLy_coarse.append([i, yValue.cpu().item()])

    aLy_coarse = torch.from_numpy(np.array(aLy_coarse)).to(ygrad_coord)

    # Since the last one is an extra point very close, I'm making it the same
    aLy_coarse[-1, 1] = aLy_coarse[-2, 1]

    # Boundary condition at point moved by gridPointsAllowed
    y_bc = torch.from_numpy(interpolation_function([x_coarse[-1]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # Boundary condition at point (ACTUAL THAT I WANT to keep fixed, i.e. roa=0.8)
    y_bc_real = torch.from_numpy(interpolation_function([x_coarse[-2]], x_coord, y_coord.cpu().numpy())).to(ygrad_coord)

    # **********************************************************************************************************
    # Define profile_constructor functions
    # **********************************************************************************************************

    def profile_constructor_coarse(x, y, multiplier=multiplier_quantity):
        """
        Construct curve in a coarse grid
        ----------------------------------------------------------------------------------------------------
        This constructs a curve in any grid, with any batch given in y=y.
        Useful for surrogate evaluations. Fast in a coarse grid. For HF evaluations,
        I need to do in a finer grid so that it is consistent with TGYRO.
        x, y must be (batch, radii),	y_bc must be (1)
        """
        return x, integrator_function(x, y, y_bc_real) / multiplier

    def profile_constructor_middle(x, y, multiplier=multiplier_quantity):
        """
        Deparamterizes a finer profile based on the values in the coarse.
        Reason why something like this is not used for the full profile is because derivative of this will not be as original,
                which is needed to match TGYRO
        """
        yCPs = CALCtools.Interp1d_torch()(aLy_coarse[:, 0][:-1].repeat((y.shape[0], 1)), y, x)
        return x, integrator_function(x, yCPs, y_bc_real) / multiplier

    def profile_constructor_fine(x, y, multiplier=multiplier_quantity):
        """
        Notes:
            - x is a 1D array, but y can be a 2D array for a batch of individuals: (batch,x)
            - I am assuming it is 1/LT for parameterization, but gives T
        """

        y = torch.atleast_2d(y)
        x = x[0, :] if x.dim() == 2 else x

        # Add the extra trick point
        x = torch.cat((x, aLy_coarse[-1][0].repeat((1))))
        y = torch.cat((y, aLy_coarse[-1][-1].repeat((y.shape[0], 1))), dim=1)

        # Model curve (basically, what happens in between points)
        yBS = CALCtools.Interp1d_torch()(x.repeat(y.shape[0], 1), y, x_notrail.repeat(y.shape[0], 1))

        """
        ---------------------------------------------------------------------------------------------------------
            Trick 1: smoothAroundCoarsing
                TGYRO will use a 2nd order scheme to obtain gradients out of the profile, so a piecewise linear
                will simply not give the right derivatives.
                Here, this rough trick is to modify the points in gradient space around the coarse grid with the
                same value of gradient, so in principle it doesn't matter the order of the derivative.
        """
        num_around = 1
        for i in range(x.shape[0] - 2):
            ir = torch.argmin(torch.abs(x[i + 1] - x_notrail))
            for k in range(-num_around, num_around + 1, 1):
                yBS[:, ir + k] = yBS[:, ir]
        # --------------------------------------------------------------------------------------------------------

        yBS = integrator_function(x_notrail.repeat(yBS.shape[0], 1), yBS.clone(), y_bc)

        """
        Trick 2: Correct y_bc
            The y_bc for the profile integration started at gridPointsAllowed, but that's not the real
            y_bc. I want the temperature fixed at my first point that I actually care for.
            Here, I multiply the profile to get that.
            Multiplication works because:
                1/LT = 1/T * dT/dr
                1/LT' = 1/(T*m) * d(T*m)/dr = 1/T * dT/dr = 1/LT
            Same logarithmic gradient, but with the right boundary condition

        """
        ir = torch.argmin(torch.abs(x_notrail - x[-2]))
        yBS = yBS * torch.transpose((y_bc_real / yBS[:, ir]).repeat(yBS.shape[1], 1), 0, 1)

        # Add trailing edge
        y_trailnew = copy.deepcopy(y_trail).repeat(yBS.shape[0], 1)

        x_notrail_t = torch.cat((x_notrail[:-1], x_trail), dim=0)
        yBS = torch.cat((yBS[:, :-1], y_trailnew), dim=1)

        return x_notrail_t, yBS / multiplier

    # **********************************************************************************************************

    return (
        aLy_coarse,
        profile_constructor_fine,
        profile_constructor_coarse,
        profile_constructor_middle,
    )


# -------------------------
# Base parameter model
# -------------------------

class ParameterBase:
    """Abstract base class for parameterizing a scalar profile y(x).

    Expected common methods:
    - get_aLy(x_eval, params) -> np.ndarray: return a/Ly on x_eval
    - get_y(x_eval, params) -> np.ndarray: return y on x_eval
    - get_curvature(x_eval, params) -> np.ndarray: return d2y/dx2 on x_eval
    - _build_interpolator(x_data, y_data) -> store interpolator
    - _build_bc_dict(boundary_model, state) -> store dict of BC values from boundary model
    - update_all(boundary_model, state) -> recalculate attributes if dirty flag is set
    
    Attributes:
    - options: Dict[str, Any] of model options
    - interpolator: callable interpolator object
    - bcs: Dict[str, float] of boundary condition values 
    - params: Dict[str, np.ndarray] of model parameters
    - y: Dict[str, np.ndarray] of reconstructed profiles
    - aLy: Dict[str, np.ndarray] of a/Ly profiles
    - dirty: bool flag indicating if model needs re-initialization

    Notes
    -----
    - x_eval is 1D normalized radius in roa (x = r/a).

    """

    def __init__(self, options: Dict[str, Any]):
        options = dict(options or {})

        self.predicted_profiles = list(options.get("predicted_profiles", []))
        self.include_zero_grad_on_axis = bool(options.get("include_zero_grad_on_axis", True))
        self.sigma = float(options.get("sigma", 0.05))
        self.bounds = options.get("bounds", None)

        self.params: Dict[str, np.ndarray] = {}
        self.param_std: Dict[str, np.ndarray] = {}
        self.param_names: List[str] = []
        self.bc_dict: Dict[str, List[BCEntry]] = {}
        self.bc_tensors: Dict[str, Dict[str, Any]] = {}

        self.lcfs_aLti_in_params = options.get('lcfs_aLti_in_params', False)
        if self.lcfs_aLti_in_params:
            #raise NotImplementedError("lcfs_aLti_in_params=True not implemented for SplineParameterModel.")
            self.param_names.append('aLti_lcfs')

    # ------------------------------
    # Abstract-like interface
    # ------------------------------
    def add_bc(self, key: str, bc: BCEntry):
        self.bc_dict.setdefault(key, []).append({"val": float(bc['val']), "loc": float(bc['loc'])})

    def build_bcs(self, bc_dict: Dict[str, Any]) -> Dict[str, List[BCEntry]]:
        """
        Normalize and store BCs. Accepts:
          bc_dict = {"ne": (1e19, 1.0), "aLne": {"val": -2.0, "loc": 1.0},
                     "aLne": [( -1.5, 0.95), (-2.0, 1.0 )] }
        Stored form: self.bc_dict['aLne'] = [ {'val':..., 'loc':...}, {...} ]
        """
        self.bc_dict = {}

        for key, val in bc_dict.items():
            # allow list of entries
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple, dict)):
                for v in val:
                    self.add_bc(key, _normalize_single_bc(v))
            else:
                # single entry (tuple/list or dict)
                self.add_bc(key, _normalize_single_bc(val))

        # Optionally ensure aL<prof> has an axis BC at 0.0 (append only if no exact axis entry)
        if self.include_zero_grad_on_axis:
            for prof in self.predicted_profiles:
                key = f"aL{prof}"
                entries = self.bc_dict.get(key, [])
                has_axis = any(np.isclose(e['loc'], 0.0) for e in entries)
                if not has_axis:
                    # append axis zero-gradient BC (do not overwrite existing BCs)
                    self.add_bc(key, {"val": 0.0, "loc": 0.0})

        return self.bc_dict

    def get_nearest_bc(self, key: str, location: float) -> Union[BCEntry, None]:
        """Return the BC entry with location nearest to the requested location."""
        entries = self.bc_dict.get(key, [])
        if not entries:
            return None
        locs = np.array([e['loc'] for e in entries], dtype=float)
        idx = int(np.argmin(np.abs(locs - location)))
        return entries[idx]

    @staticmethod
    def _to_numpy_any(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _is_batched_bc_input(self, bc_dict: Dict[str, Any], batch_size: int) -> bool:
        if not isinstance(bc_dict, dict):
            return False
        for val in bc_dict.values():
            if not (isinstance(val, dict) and ("val" in val) and ("loc" in val)):
                continue
            v = self._to_numpy_any(val["val"])
            l = self._to_numpy_any(val["loc"])
            if v.ndim >= 1 and l.ndim >= 1 and v.shape[0] == batch_size and l.shape[0] == batch_size:
                return True
        return False

    def _slice_batched_bc_dict(self, bc_dict: Dict[str, Any], batch_idx: int, batch_size: int) -> Dict[str, Any]:
        """Convert batched BC tensors into a scalar/list BC dict for one batch index."""
        sliced: Dict[str, Any] = {}

        for key, val in bc_dict.items():
            if not (isinstance(val, dict) and ("val" in val) and ("loc" in val)):
                sliced[key] = val
                continue

            v = self._to_numpy_any(val["val"])
            l = self._to_numpy_any(val["loc"])

            if not (v.ndim >= 1 and l.ndim >= 1 and v.shape[0] == batch_size and l.shape[0] == batch_size):
                sliced[key] = val
                continue

            if "mask" in val:
                m = self._to_numpy_any(val["mask"]).astype(bool)
            else:
                m = np.ones_like(v, dtype=bool)

            v_row = np.ravel(v[batch_idx])
            l_row = np.ravel(l[batch_idx])
            m_row = np.ravel(m[batch_idx]) if m.ndim >= 1 and m.shape[0] == batch_size else np.ones_like(v_row, dtype=bool)

            entries = []
            for j in range(min(len(v_row), len(l_row), len(m_row))):
                if m_row[j]:
                    entries.append((float(v_row[j]), float(l_row[j])))

            if len(entries) == 1:
                sliced[key] = entries[0]
            elif len(entries) > 1:
                sliced[key] = entries

        return sliced
    
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract parameters from a PlasmaState given boundary conditions."""
        raise NotImplementedError

    def get_aLy(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute scale length a/Ly = -a * (dy/dx) / y for each profile."""
        raise NotImplementedError

    def get_y(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profile y(x) from parameter set."""
        raise NotImplementedError

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² for each profile."""
        raise NotImplementedError

    def update(self, params: Dict[str, np.ndarray], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        """Convenience method returning y, aLy, and curvature."""
        self.bc_tensors = {}

        def _to_numpy(value: Any) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                # Support torch tensors without requiring torch as a hard dependency.
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def _slice_params_for_batch(
            all_params: Dict[str, Any], batch_idx: int, batch_size: int
        ) -> Dict[str, Any]:
            sliced: Dict[str, Any] = {}
            for prof, prof_params in all_params.items():
                if isinstance(prof_params, dict):
                    prof_sliced: Dict[str, Any] = {}
                    for name, value in prof_params.items():
                        arr = _to_numpy(value)
                        # Dict-valued parameters in these models are scalar per profile;
                        # a leading batch axis indicates per-batch values.
                        if arr.ndim == 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = float(arr[batch_idx])
                        elif arr.ndim > 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = arr[batch_idx]
                        else:
                            prof_sliced[name] = value
                    sliced[prof] = prof_sliced
                else:
                    arr = _to_numpy(prof_params)
                    # For array-valued params, only treat as batched when a leading
                    # batch dimension is explicit (ndim > 1).
                    if arr.ndim > 1 and arr.shape[0] == batch_size:
                        sliced[prof] = arr[batch_idx]
                    else:
                        sliced[prof] = prof_params
            return sliced

        x_arr = _to_numpy(x_eval)

        if x_arr.ndim <= 1:
            self.build_bcs(bc_dict)
            y = self.get_y(params, x_arr)
            aLy = self.get_aLy(params, x_arr)
            curvature = self.get_curvature(params, x_arr)
            return y, aLy, curvature

        # Batched evaluation: iterate over batch dimension and stack results.
        batch_size = x_arr.shape[0]
        use_batched_bcs = self._is_batched_bc_input(bc_dict, batch_size)
        if use_batched_bcs:
            self.bc_tensors = bc_dict
        else:
            self.build_bcs(bc_dict)

        y_batches: Dict[str, List[np.ndarray]] = {}
        aLy_batches: Dict[str, List[np.ndarray]] = {}
        curv_batches: Dict[str, List[np.ndarray]] = {}

        for i in range(batch_size):
            if use_batched_bcs:
                self.build_bcs(self._slice_batched_bc_dict(bc_dict, i, batch_size))
            batch_params = _slice_params_for_batch(params, i, batch_size)
            y_i = self.get_y(batch_params, x_arr[i])
            aLy_i = self.get_aLy(batch_params, x_arr[i])
            curv_i = self.get_curvature(batch_params, x_arr[i])

            for prof, vals in y_i.items():
                y_batches.setdefault(prof, []).append(_to_numpy(vals))
            for prof, vals in aLy_i.items():
                aLy_batches.setdefault(prof, []).append(_to_numpy(vals))
            for prof, vals in curv_i.items():
                curv_batches.setdefault(prof, []).append(_to_numpy(vals))

        y_out = {prof: np.stack(vals, axis=0) for prof, vals in y_batches.items()}
        aLy_out = {prof: np.stack(vals, axis=0) for prof, vals in aLy_batches.items()}
        curv_out = {prof: np.stack(vals, axis=0) for prof, vals in curv_batches.items()}

        self.y = y_out
        self.aLy = aLy_out
        self.curv = curv_out
        return y_out, aLy_out, curv_out


# -------------------------
# Spline parameter model
# -------------------------


class Spline(ParameterBase):
    """Spline-based parameterization of a/Ly with control points at user-defined knots.

    Design parameters: self.defined_on + i for i in range(len(knots))

    Parameters (options)
    --------------------
    knots : Sequence[float]
        Locations in x (roa=r/a) where parameters define a/Ly values.
    spline_type : str
        'akima' (default) or 'pchip'. Determines the interpolator.
    include_zero_grad_on_axis : bool
        If True (default) and knots do not include x=0, a virtual control point with a/Ly=0 at x=0
        is prepended for smooth behavior at the magnetic axis.
    bc_field : Optional[str]
        Name of boundary condition value on state.BC (e.g., 'ne', 'te', 'ti') to use as y_sep if
        not explicitly provided to y()/curvature().
    """

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.spline_type = options.get('spline_type', 'linear').lower()
        self.knots = np.array(options.get('knots', []))
        self.defined_on = options.get('defined_on', 'aLy')
        if self.spline_type not in ('akima', 'pchip', 'cubic', 'linear'):
            raise ValueError("spline_type must be 'akima', 'pchip', 'cubic', or 'linear'")
        self.param_names = [self.defined_on+str(i) for i in range(len(self.knots))]
        self.n_params_per_profile = len(self.knots)
        self.splines: Dict[str, Any] = {}
        self._trailing_edge: Dict[str, Any] = {}

    # ------------------------------
    # Internal utilities
    # ------------------------------
    def _make_spline(self, x: np.ndarray, y: np.ndarray, prof: str):
        """Return a spline object of chosen type.
        
        If include_zero_grad_on_axis=True and x doesn't start at 0, prepend axis BC.
        """
        x_spline = np.asarray(x)
        y_spline = np.asarray(y)

        if self.include_zero_grad_on_axis and not np.isclose(x_spline[0], 0.0) and self.defined_on == 'aLy':
            x_spline = np.insert(x_spline, 0, 0.0)
            y_spline = np.insert(y_spline, 0, 0.0)
        
        # Build spline
        if self.spline_type == "akima":
            spline = akima(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "pchip":
            spline = pchip(x_spline, y_spline, extrapolate=True)
        elif self.spline_type in ("cubic", "cspline"):
            spline = CubicSpline(x_spline, y_spline, extrapolate=True)
        elif self.spline_type == "linear":
            spline = linear(x_spline, y_spline, k=1)
        else:
            raise ValueError(f"Unknown spline_type: {self.spline_type}")
        
        self.splines[prof] = spline
        return spline
    
    def _get_spline(self, prof: str):
        """Retrieve a cached spline."""
        spline = self.splines.get(prof)
        if spline is None:
            raise KeyError(f"Spline for profile '{prof}' not initialized.")
        return spline

    def _integrate_aLy(self, prof: str, x_eval: np.ndarray, spl: Any, bc_value: float, bc_loc: float) -> np.ndarray:
        """
        Integrate spline of a/Ly to recover y(x) via
            dy/dx = -aLy * y   (for x = roa = r/a dimensionless)

            => y(x) = y_bc * exp(-∫[bc_loc to x] aLy dx')

        For bc_loc = 1.0 (edge), integrating inward (decreasing x) gives positive integral.
        For bc_loc = 0.0 (axis), integrating outward (increasing x) gives negative integral.
        """

        if not hasattr(spl, "antiderivative"):
            raise TypeError(f"Spline type {type(spl)} has no .antiderivative()")

        # y(x) = y_bc * exp(-∫[bc_loc→x] aLy dx')
        # aLy = a/Ly = -d(ln y)/d(roa), so the integral is already dimensionless.
        F = spl.antiderivative()
        phase = -(F(x_eval) - F(bc_loc))

        return bc_value * np.exp(phase)

    # ------------------------------
    # Implement required methods
    # ------------------------------
    def parameterize(self, state, bc_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract spline coefficients (y values at knots) from a state.
        
        Incorporates boundary conditions by merging BC points into the spline
        construction data before extracting parameters at knots.
        """

        if self.bounds is None:
            self.bounds = {name: (0.0, 100.0) for name in self.param_names}
        
        self.a = state.a  # store for conversions

        self.build_bcs(bc_dict)
        params = {}
        roa_vals = getattr(state, 'roa')
        
        for prof in self.predicted_profiles:
            prof_name = f"aL{prof}" if self.defined_on == "aLy" else prof
            y_prof = getattr(state, prof_name)
            if y_prof.ndim == 2:
                y_prof = y_prof[:, 0].flatten()
            else:
                y_prof = np.asarray(y_prof).flatten()
            
            # Merge boundary conditions into spline data
            x_data = np.asarray(roa_vals).flatten()
            y_data = np.asarray(y_prof).flatten()
            
            # Get BC entries for this profile
            bc_entries = self.bc_dict.get(prof_name, [])
            
            #Add/replace BC points in the data
            for bc in bc_entries:
                bc_loc = bc['loc']
                bc_val = bc['val']
                
                #Find if this location already exists in data (within tolerance)
                existing_idx = np.where(np.isclose(x_data, bc_loc, atol=1e-6))[0]
                
                if len(existing_idx) > 0:
                    # Replace existing point
                    y_data[existing_idx[0]] = bc_val
                else:
                    # Insert new point in sorted order
                    insert_idx = np.searchsorted(x_data, bc_loc)
                    x_data = np.insert(x_data, insert_idx, bc_loc)
                    y_data = np.insert(y_data, insert_idx, bc_val)
            
            # Store trailing edge data (from last knot onward) for bc=None fallback in get methods
            last_knot = self.knots[-1]
            split = np.searchsorted(x_data, last_knot, side='right')
            te_start = max(0, split - 1)  # include one overlap point at/before last_knot
            te_x = x_data[te_start:]
            def _te_arr(attr):
                try:
                    arr = np.asarray(getattr(state, attr)).flatten()
                    return arr[te_start:]
                except AttributeError:
                    return None
            self._trailing_edge[prof] = {
                'x':   te_x,
                'y':   _te_arr(prof),
                'aLy': _te_arr(f'aL{prof}'),
            }

            # Build spline with BC-augmented data
            spline = self._make_spline(x_data, y_data, prof)
            params[prof] = dict(zip(self.param_names, spline(self.knots)))
        
        self.params = params
        std_dict = {prof: {name: abs(val)*self.sigma for name, val in params[prof].items()} for prof in params}
        self.param_std = std_dict

        return params, std_dict  # return nominal and std dev
    
    def get_y(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute profiles y(x) on x_eval."""
        out = {}
        for prof, prof_params in params.items():
            
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            use_ghost = False
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                # Ghost point: add a point just beyond the last knot with the same value;
                # from there to roa=1 y is held constant (trailing edge stitching).
                _eps = 1e-4
                knots = np.append(self.knots, self.knots[-1] + _eps)
                vals = np.append(vals, vals[-1])
                use_ghost = True

            ghost_knot = knots[-1]
            spline = self._make_spline(knots, vals, prof)

            if self.defined_on == "y":
                y_spl = spline(x_eval)
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_y = te.get('x'), te.get('y')
                    if te_x is not None and te_y is not None and len(te_x) >= 2:
                        y_trail = np.interp(x_eval, te_x, te_y)
                    else:
                        y_trail = np.full_like(x_eval, float(spline(self.knots[-1])))
                    y = np.where(x_eval > self.knots[-1], y_trail, y_spl)
                else:
                    y = y_spl
            elif self.defined_on == "aLy":
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' at x=1.0")
                y_full = self._integrate_aLy(prof, x_eval, spline, bc_y['val'], bc_y['loc'])
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_y = te.get('x'), te.get('y')
                    if te_x is not None and te_y is not None and len(te_x) >= 2:
                        y_trail = np.interp(x_eval, te_x, te_y)
                    else:
                        y_trail = np.full_like(x_eval, self._integrate_aLy(prof, np.array([ghost_knot]), spline, bc_y['val'], bc_y['loc'])[0])
                    y = np.where(x_eval > self.knots[-1], y_trail, y_full)
                else:
                    y = y_full
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")

            out[prof] = np.clip(y, a_min=0, a_max=None)
        self.y = out
        return out

    def get_aLy(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute a/Ly(x) on x_eval.
        
        For roa (normalized): aLy = -(dy/dx) / y where x = r/a
        """
        out = {}
        for prof, prof_params in params.items():
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)
        
            # get aLy boundary condition to update vals if needed
            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            use_ghost = False
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                # Ghost point: add a point just beyond the last knot with the same value;
                # from there to roa=1 y is held constant (aLy=0 in that region).
                _eps = 1e-4
                knots = np.append(self.knots, self.knots[-1] + _eps)
                vals = np.append(vals, vals[-1])
                use_ghost = True

            ghost_knot = knots[-1]
            spline = self._make_spline(knots, vals, prof)

            if self.defined_on == "aLy":
                aLy_spl = spline(x_eval)
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_aLy = te.get('x'), te.get('aLy')
                    if te_x is not None and te_aLy is not None and len(te_x) >= 2:
                        aLy_trail = np.interp(x_eval, te_x, te_aLy)
                    else:
                        aLy_trail = np.zeros_like(x_eval)
                    aLy = np.where(x_eval > self.knots[-1], aLy_trail, aLy_spl)
                else:
                    aLy = aLy_spl
            elif self.defined_on == "y":
                if use_ghost:
                    te = self._trailing_edge.get(prof) or {}
                    te_x, te_y = te.get('x'), te.get('y')
                    te_aLy_arr = te.get('aLy')
                    if te_x is not None and te_y is not None and len(te_x) >= 2:
                        y_trail = np.interp(x_eval, te_x, te_y)
                        if te_aLy_arr is not None:
                            aLy_trail = np.interp(x_eval, te_x, te_aLy_arr)
                        else:
                            te_spl = pchip(te_x, te_y)
                            y_t = te_spl(x_eval)
                            dy_t = te_spl.derivative(1)(x_eval)
                            y_t_safe = np.where(np.abs(y_t) < 1e-12, 1e-12, y_t)
                            aLy_trail = -dy_t / y_t_safe
                        y_spl = spline(x_eval)
                        dy_spl = spline.derivative(1)(x_eval)
                        y_spl_safe = np.where(np.abs(y_spl) < 1e-12, 1e-12, y_spl)
                        aLy_interior = -dy_spl / y_spl_safe
                        aLy = np.where(x_eval > self.knots[-1], aLy_trail, aLy_interior)
                    else:
                        y = spline(x_eval)
                        dy = spline.derivative(1)(x_eval)
                        y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                        aLy = -dy / y_safe
                else:
                    y = spline(x_eval)
                    dy = spline.derivative(1)(x_eval)
                    # aLy = a/Ly = -(dy/dx)/y  for x = r/a dimensionless
                    y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
                    aLy = -dy / y_safe
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = np.clip(aLy, a_min=0, a_max=None)
        self.aLy = out
        return out

    def get_curvature(self, params: Dict[str, np.ndarray], x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute d²y/dx² on x_eval.

        For roa (x = r/a dimensionless), aLy = -d(ln y)/dx, so:
            y' = -aLy * y
            y'' = -(aLy' * y + aLy * y') = -y * (aLy' - aLy²)
        """
        out = {}
        for prof, prof_params in params.items():
            if isinstance(prof_params, dict):
                vals = np.array([prof_params[n] for n in self.param_names])
            else:
                vals = np.asarray(prof_params)

            bc_name = f'aL{prof}' if self.defined_on == 'aLy' else prof
            bc = self.get_nearest_bc(bc_name, 1.0)
            use_ghost = False
            if bc is not None:
                # add bc point to vals and knots for spline construction
                if not np.any(np.isclose(self.knots, 1.0)):
                    knots = np.append(self.knots, bc['loc'])
                    vals = np.append(vals, bc['val'])
                else:
                    # find nearest knot to bc location
                    knot_diffs = np.abs(self.knots - bc['loc'])
                    nearest_knot_idx = int(np.argmin(knot_diffs))
                    vals[nearest_knot_idx] = bc['val']
                    knots = self.knots
            else:
                # Ghost point: add a point just beyond the last knot with the same value;
                # from there to roa=1 y is held constant (aLy=0, curvature=0 in that region).
                _eps = 1e-4
                knots = np.append(self.knots, self.knots[-1] + _eps)
                vals = np.append(vals, vals[-1])
                use_ghost = True

            ghost_knot = knots[-1]
            spline = self._make_spline(knots, vals, prof)
            te = self._trailing_edge.get(prof) or {}
            te_x, te_y = te.get('x'), te.get('y')
            _has_te = use_ghost and te_x is not None and te_y is not None and len(te_x) >= 3

            if self.defined_on == "y":
                curv_interior = spline.derivative(2)(x_eval)
                if use_ghost:
                    if _has_te:
                        curv_trail = pchip(te_x, te_y).derivative(2)(x_eval)
                    else:
                        curv_trail = np.zeros_like(x_eval)
                    curv = np.where(x_eval > self.knots[-1], curv_trail, curv_interior)
                else:
                    curv = curv_interior
            elif self.defined_on == "aLy":
                # Get boundary condition to integrate aLy → y
                bc_y = self.get_nearest_bc(prof, 1.0)
                if bc_y is None:
                    raise ValueError(f"No boundary condition found for profile '{prof}' to compute curvature")

                aLy_spl = spline
                aLy_interior = aLy_spl(x_eval)
                aLy_prime_interior = aLy_spl.derivative(1)(x_eval)
                y_interior = self._integrate_aLy(prof, x_eval, aLy_spl, bc_y['val'], bc_y['loc'])

                if use_ghost:
                    if _has_te:
                        te_curv_spl = pchip(te_x, te_y)
                        y_trail = np.interp(x_eval, te_x, te_y)
                        aLy_trail_arr = te.get('aLy')
                        if aLy_trail_arr is not None:
                            aLy_trail = np.interp(x_eval, te_x, aLy_trail_arr)
                            aLy_prime_trail = pchip(te_x, aLy_trail_arr).derivative(1)(x_eval)
                        else:
                            aLy_trail = -te_curv_spl.derivative(1)(x_eval) / np.where(np.abs(te_curv_spl(x_eval)) < 1e-12, 1e-12, te_curv_spl(x_eval))
                            aLy_prime_trail = np.gradient(aLy_trail, x_eval)
                        aLy = np.where(x_eval > self.knots[-1], aLy_trail, aLy_interior)
                        aLy_prime = np.where(x_eval > self.knots[-1], aLy_prime_trail, aLy_prime_interior)
                        y = np.where(x_eval > self.knots[-1], y_trail, y_interior)
                    else:
                        aLy = aLy_interior
                        aLy_prime = aLy_prime_interior
                        y = y_interior
                else:
                    aLy = aLy_interior
                    aLy_prime = aLy_prime_interior
                    y = y_interior

                # Avoid division issues and NaN propagation
                y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)

                # y'' = -y * (aLy' - aLy²)  for x = r/a dimensionless
                curv = -y_safe * (aLy_prime - aLy**2)

                # Clean up any remaining NaN/inf values
                curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError(f"Invalid defined_on: {self.defined_on}")
            out[prof] = curv
        self.curv = out
        return out


class Mtanh(ParameterBase):
    """Modified-tanh profile model with position-dependent width.

    Profile form:
        y(x) = A * (1 - tanh(u)) - m * (x - 1) + b
        u(x) = (x - c) / (Delta_0 * (1 + delta * (x - c)))

    Derivatives:
        dy/dx = -A * sech^2(u) / (Delta_0 * (1 + delta * (x - c))^2) - m
        a/Ly  = -(dy/dx) / y

    Solver-space parameters are unconstrained logs:
        [log_alpha, log_Delta_0, log_u1, log_Ralpha]
    with:
        alpha = A / Delta_0 > 0
        u1 = u(1) > 0
        R = y(1) * aLy(1) - m > 0
        log_Ralpha = log(R / alpha)

    Boundary conditions y(1), aLy(1) are enforced algebraically (no root-find):
        A = alpha * Delta_0
        m = y(1) * aLy(1) - R
        W1_eff = sqrt(A * sech^2(u1) / R)
        s = u1 * W1_eff
        c = 1 - s
        delta = (W1_eff / Delta_0 - 1) / s
        b = y(1) - A * (1 - tanh(u1))
    """

    _WIDTH_FLOOR = 1e-6
    _Y_FLOOR = 1e-12

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.defined_on = "y"

        self.param_names = [
            'log_alpha',
            'log_Delta_0',
            'log_u1',
            'log_Ralpha',
        ]

        self.n_params_per_profile = len(self.param_names)
        self.include_zero_grad_on_axis = False

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _to_physical(
        self,
        p: Union[np.ndarray, Dict[str, float]],
        y_bc: float,
        aLy_bc: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Map solver-space parameters to physical mtanh parameters.

        Parameters
        ----------
        p
            Solver-space parameter vector or dict:

                [log_alpha, log_Delta_0, log_u1, log_Ralpha]

        y_bc
            Boundary value y(1).

        aLy_bc
            Boundary value a/Ly(1).

        Returns
        -------
        A, Delta_0, delta, m, c, b
            Fully reconstructed physical parameter set satisfying the
            boundary conditions exactly.
        """
        if isinstance(p, dict):
            vec = np.array([p[n] for n in self.param_names], dtype=float)
        else:
            vec = np.asarray(p, dtype=float)

        log_alpha, log_D0, log_u1, log_Ralpha = vec

        alpha   = float(np.exp(log_alpha))
        Delta_0 = float(np.exp(log_D0))
        u1      = float(np.exp(log_u1))

        R = float(alpha * np.exp(log_Ralpha))

        A = alpha * Delta_0
        m = y_bc * aLy_bc - R

        th1     = np.tanh(u1)
        sech2_1 = 1.0 - th1**2

        W1_eff = np.sqrt(A * sech2_1 / R)

        s = u1 * W1_eff
        c = 1.0 - s

        if abs(s) < 1e-12:
            delta = 0.0
        else:
            delta = (W1_eff / Delta_0 - 1.0) / s

        b = y_bc - A * (1.0 - th1)

        return A, Delta_0, delta, m, c, b

    def _physical_to_solver(
        self,
        A: float,
        Delta_0: float,
        delta: float,
        m: float,
        c: float,
        y_bc: float,
        aLy_bc: float,
    ) -> Dict[str, float]:
        """Map physical mtanh parameters to solver space.

        Parameters
        ----------
        A, Delta_0, delta, m, c
            Physical mtanh parameters.

        y_bc
            Boundary value y(1).

        aLy_bc
            Boundary value a/Ly(1).

        Returns
        -------
        dict
            Solver-space parameter dictionary.
        """
        alpha = A / Delta_0
        s = 1.0 - c
        u1 = s / (Delta_0 * (1.0 + delta * s))
        R = y_bc * aLy_bc - m

        return {
            'log_alpha':   float(np.log(alpha)),
            'log_Delta_0': float(np.log(Delta_0)),
            'log_u1':      float(np.log(u1)),
            'log_Ralpha':  float(np.log(R / alpha)),
        }

    def _resolve_bcs(self, prof: str) -> Tuple[float, float]:
        """Return (y_bc, aLy_bc) at x=1 for the given profile."""
        bc_y   = self.get_nearest_bc(prof, 1.0)
        bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
        if bc_y is None:
            raise ValueError(f"No y BC found for profile '{prof}'")
        if bc_aLy is None:
            raise ValueError(f"No aLy BC found for profile '{prof}'")
        return float(bc_y['val']), float(bc_aLy['val'])

    def _w_eval(self, x: np.ndarray, Delta_0: float, delta: float, c: float) -> np.ndarray:
        """Spatially varying width w(x) = Delta_0*(1+delta*(x-c)), floored at 1e-6."""
        return np.maximum(Delta_0 * (1.0 + delta * (x - c)), self._WIDTH_FLOOR)

    def _y_eval(self, x: np.ndarray, A: float, Delta_0: float, delta: float,
                m: float, c: float, b: float) -> np.ndarray:
        w  = self._w_eval(x, Delta_0, delta, c)
        u  = (x - c) / w
        return A * (-np.tanh(u) + 1.0) - m * (x - 1.0) + b

    def _dydx_eval(self, x: np.ndarray, A: float, Delta_0: float, delta: float,
                   m: float, c: float) -> np.ndarray:
        """dy/dx = -A * sech^2(u) / (Delta_0*(1+delta*(x-c))^2) - m"""
        f     = np.maximum(1.0 + delta * (x - c), self._WIDTH_FLOOR)
        u     = (x - c) / (Delta_0 * f)
        sech2 = 1.0 - np.tanh(u) ** 2
        return -A * sech2 / (Delta_0 * f ** 2) - m

    def _physical_for_profile(
        self,
        prof: str,
        p: Union[np.ndarray, Dict[str, float]],
    ) -> Tuple[float, float, float, float, float, float]:
        """Resolve BCs for a profile and map solver-space params to physical params."""
        y_bc, aLy_bc = self._resolve_bcs(prof)
        return self._to_physical(p, y_bc, aLy_bc)

    # ─────────────────────────────────────────────────────────
    # ParameterBase interface
    # ─────────────────────────────────────────────────────────

    def parameterize(
        self,
        state: Any,
        bc_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Fit the mtanh model to profile data and store solver-space parameters.

        The fitted solver-space parameters are:

            phi = {
                log_alpha,
                log_Delta_0,
                log_u1,
                log_Ralpha,
            }

        Boundary conditions y(1) and a/Ly(1) are enforced analytically through
        reconstruction of the derived physical parameters:

            A, Delta_0, delta, m, c, b

        so the optimizer operates entirely in unconstrained solver space.
        """
        self.build_bcs(bc_dict)

        params: Dict[str, Dict[str, float]] = {}
        params_std: Dict[str, Dict[str, float]] = {}

        x_data = np.asarray(getattr(state, 'roa')).flatten()

        for prof in self.predicted_profiles:

            y_data = np.asarray(getattr(state, prof)).flatten()

            y_bc, aLy_bc = self._resolve_bcs(prof)

            def _model(
                x,
                log_alpha,
                log_D0,
                log_u1,
                log_Ralpha,
                _y_bc=y_bc,
                _aLy_bc=aLy_bc,
            ):
                p = np.array([log_alpha, log_D0, log_u1, log_Ralpha])
                A_, D0_, delta_, m_, c_, b_ = self._to_physical(p, _y_bc, _aLy_bc)
                R = float(np.exp(log_alpha) * np.exp(log_Ralpha))
                if m_ <= 0 or R >= _y_bc * _aLy_bc:
                    return np.full_like(x, fill_value=1e3)  # large penalty for unphysical parameters
                return self._y_eval(x, A_, D0_, delta_, m_, c_, b_)

            # ─────────────────────────────────────────────────────
            # Heuristic initial guess
            # ─────────────────────────────────────────────────────

            alpha0   = 1.0
            D00      = 0.05
            u10      = 2.0
            Ralpha0  = 1.0

            p0 = np.array([np.log(alpha0),np.log(D00),np.log(u10),np.log(Ralpha0)])

            # Optional broad bounds for optimizer stability only
            blo = np.array([-7.0, -5.0, -7.0, -20.0])
            bhi = np.array([ 7.0, 0.0, 5.0, 20.0])

            try:

                popt, _ = curve_fit(
                    _model,
                    x_data,
                    y_data,
                    p0=p0,
                    bounds=(blo, bhi),
                    maxfev=4000,
                    xtol=1e-10,
                    ftol=1e-10,
                )

            except Exception:

                popt = p0

            # Reconstruct physical parameters once to ensure consistency
            A_fit, D0_fit, delta_fit, m_fit, c_fit, _ = self._to_physical(
                popt,
                y_bc,
                aLy_bc,
            )

            params[prof] = self._physical_to_solver(
                A_fit,
                D0_fit,
                delta_fit,
                m_fit,
                c_fit,
                y_bc,
                aLy_bc,
            )

            params_std[prof] = {
                k: abs(v) * self.sigma
                for k, v in params[prof].items()
            }

        self.params = params
        self.params_std = params_std

        return params, params_std


    def get_y(
        self,
        params: Dict[str, Dict[str, float]],
        x_eval: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate y(x) for each profile."""
        x_eval = np.asarray(x_eval)

        out: Dict[str, np.ndarray] = {}

        for prof, p in params.items():
            A, Delta_0, delta, m, c, b = self._physical_for_profile(prof, p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            R = float(np.exp(p[0]) * np.exp(p[3]))
            
            if m <= 0 or R >= y_bc * aLy_bc:
                out[prof] = np.full_like(x_eval, fill_value=1e3)  # large penalty for unphysical parameters
            else:
                out[prof] = np.clip(
                    self._y_eval(x_eval, A, Delta_0, delta, m, c, b),
                    0.0,
                    None,
                )

        self.y = out
        return out

    def get_aLy(
        self,
        params: Dict[str, Dict[str, float]],
        x_eval: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate a/Ly(x) = -(dy/dx)/y for each profile."""
        x_eval = np.asarray(x_eval)

        out: Dict[str, np.ndarray] = {}

        for prof, p in params.items():
            A, Delta_0, delta, m, c, b = self._physical_for_profile(prof, p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            R = float(np.exp(p[0]) * np.exp(p[3]))

            if m <= 0 or R >= y_bc * aLy_bc:
                out[prof] = np.full_like(x_eval, fill_value=1e3)  # large penalty for unphysical parameters
            else:
                y = self._y_eval(x_eval, A, Delta_0, delta, m, c, b)
                dydx = self._dydx_eval(x_eval, A, Delta_0, delta, m, c)
                y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)

                out[prof] = np.clip(-dydx / y_safe, 0.0, None)

        self.aLy = out
        return out

    def get_curvature(
        self,
        params: Dict[str, Dict[str, float]],
        x_eval: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Evaluate d²y/dx² on x_eval for each profile.

        With:

            w(x) = Delta_0 * (1 + delta*(x-c))
            f(x) = 1 + delta*(x-c)
            u(x) = (x-c) / (Delta_0*f(x))

        the curvature is:

            d²y/dx² =
                (2*A*sech²(u)/Delta_0)
                * [tanh(u)/(Delta_0*f^4) + delta/f^3]
        """
        x_eval = np.asarray(x_eval)

        out: Dict[str, np.ndarray] = {}

        for prof, p in params.items():
            A, Delta_0, delta, m, c, _ = self._physical_for_profile(prof, p)
            y_bc, aLy_bc = self._resolve_bcs(prof)
            R = float(np.exp(p[0]) * np.exp(p[3]))

            if m <= 0 or R >= y_bc * aLy_bc:
                out[prof] = np.full_like(x_eval, fill_value=1e3)  # large penalty for unphysical parameters
            else:
                f = np.maximum(1.0 + delta * (x_eval - c), self._WIDTH_FLOOR)

                u = (x_eval - c) / (Delta_0 * f)

                tanh_ = np.tanh(u)
                sech2 = 1.0 - tanh_**2

                out[prof] = (
                    (2.0 * A * sech2 / Delta_0)
                    * (
                        tanh_ / (Delta_0 * f**4)
                        + delta / f**3
                    )
                )

        self.curv = out
        return out

class SplineMtanh(ParameterBase):
    """Hybrid spline-like optimizer interface with mtanh profile construction.

    SplineMtanh design variables remain knot-local (either `dy*` or `aLy*`),
    but Mtanh reconstruction now uses an explicit reparameterization with free
    variables `{A, u1, delta, m}` represented in solver space as
    `{log_A, log_u1, delta, m}`.

    From boundary conditions `(y(1), aLy(1))` and free variables, derived terms
    are computed algebraically: `c` is solved deterministically from the
    quadratic constraint, and `Delta_0` and `b` follow directly.

    Fitting uses a single path everywhere in this class:
    2-point multistart (with optional cached warm-start) followed by bounded
    `least_squares(..., method='trf')` with smooth feasibility penalties. `update()`
    resolves each profile once and reuses that result for y, aLy, and curvature.
    """

    _WIDTH_FLOOR = 1e-6
    _Y_FLOOR = 1e-12
    _PENALTY = 1e3
    _A_BOUNDS = (1e-3, 5.0)
    _U1_BOUNDS = (1e-3, 10.0)
    _DELTA_BOUNDS = (-50.0, 50.0)
    _D0_BOUNDS = (1e-2, 1.0)
    _C_BOUNDS = (0.9, 1.0)

    def __init__(self, options: Dict[str, Any]):
        super().__init__(options)
        self.include_zero_grad_on_axis = False
        self.knots = np.array(options.get('knots', []) or [], dtype=float)
        self.defined_on = str(options.get('defined_on', 'aLy'))
        if self.defined_on not in ('dy', 'aLy'):
            raise ValueError("SplineMtanh defined_on must be 'dy' or 'aLy'")

        self.fit_max_nfev = int(options.get('fit_max_nfev', 100))
        self.fit_penalty_weight = float(options.get('fit_penalty_weight', 1e3))
        self.fit_feasibility_tol = float(options.get('fit_feasibility_tol', 1e-3))
        # TRF fitter tolerances.
        self.fit_trf_ftol = float(options.get('fit_trf_ftol', 1e-3))
        self.fit_trf_gtol = float(options.get('fit_trf_gtol', 1e-12))
        self.fit_trf_xtol = float(options.get('fit_trf_xtol', 1e-3))
        self.fit_trf_diff_step = float(options.get('fit_trf_diff_step', 5e-2))
        # Single pointwise acceptance criterion: max relative error across all knot points.
        self.fit_max_rel_error = float(options.get('fit_max_rel_error', 2.5e-2))
        # Resolve cache: skip re-fitting when params+BCs are within tolerance of a prior call.
        self.fit_cache_enabled = bool(options.get('fit_cache_enabled', True))
        self.fit_cache_tol = float(options.get('fit_cache_tol', 1e-1))
        self.fit_cache_max_size = int(options.get('fit_cache_max_size', 512))
        # Each entry: (cache_key, solver_theta, phys) — theta stored so IDW interpolation
        # happens in solver space and BCs can be reprojected correctly for each query.
        self._resolve_cache: Dict[str, List[Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]]] = {}
        self._last_theta_guess: Dict[str, np.ndarray] = {}

        prefix = 'dy' if self.defined_on == 'dy' else 'aLy'
        self.param_names = [f'{prefix}{i}' for i in range(len(self.knots))]
        self.n_params_per_profile = len(self.param_names)

    @staticmethod
    def _f(x, c: float, delta: float) -> np.ndarray:
        return np.maximum(1.0 + delta * (np.asarray(x, dtype=float) - c), SplineMtanh._WIDTH_FLOOR)

    @staticmethod
    def _y_mtanh(x, A: float, Delta_0: float, delta: float,
                 m: float, c: float, b: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        f = SplineMtanh._f(x, c, delta)
        u = (x - c) / (Delta_0 * f)
        return A * (1.0 - np.tanh(u)) - m * (x - 1.0) + b

    @staticmethod
    def _dydx_mtanh(x, A: float, Delta_0: float, delta: float,
                    m: float, c: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        f = SplineMtanh._f(x, c, delta)
        u = (x - c) / (Delta_0 * f)
        sech2 = 1.0 - np.tanh(u) ** 2
        return -A * sech2 / (Delta_0 * f ** 2) - m

    @staticmethod
    def _d2ydx2_mtanh(x, A: float, Delta_0: float, delta: float,
                      c: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        f = SplineMtanh._f(x, c, delta)
        u = (x - c) / (Delta_0 * f)
        tanh_ = np.tanh(u)
        sech2 = 1.0 - tanh_ ** 2
        return (2.0 * A * sech2 / Delta_0) * (
            tanh_ / (Delta_0 * f ** 4) + delta / f ** 3
        )

    @staticmethod
    def _penalty_vec(n: int) -> np.ndarray:
        return np.full(int(max(1, n)), SplineMtanh._PENALTY, dtype=float)

    @staticmethod
    def _solve_c(delta: float, target: float) -> Optional[float]:
        if abs(delta) < 1e-10:
            c = 1.0 - target
        else:
            discriminant = 1.0 + 4.0 * delta * target
            if discriminant < 0.0:
                return None
            root1 = (-1.0 + np.sqrt(discriminant)) / (2.0 * delta)
            root2 = (-1.0 - np.sqrt(discriminant)) / (2.0 * delta)
            valid = [r for r in (root1, root2) if r > 0.0]
            if not valid:
                return None
            one_minus_c = min(valid)
            c = 1.0 - one_minus_c

        if not (0.0 < c < 1.0):
            return None
        return float(c)

    def _to_physical(
        self,
        p: Union[np.ndarray, Dict[str, float]],
        y_bc: float,
        aLy_bc: float,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        phys, violation = self._to_physical_projected(p, y_bc, aLy_bc)
        if np.max(np.abs(violation)) > self.fit_feasibility_tol:
            return None
        return phys

    def _to_physical_projected(
        self,
        p: Union[np.ndarray, Dict[str, float]],
        y_bc: float,
        aLy_bc: float,
    ) -> Tuple[Tuple[float, float, float, float, float, float], np.ndarray]:
        if isinstance(p, dict):
            vec = np.array([p['log_A'], p['log_u1'], p['delta'], p['m']], dtype=float)
        else:
            vec = np.asarray(p, dtype=float).reshape(-1)

        log_A, log_u1, delta, m = [float(v) for v in vec[:4]]

        A  = np.exp(log_A)
        u1 = np.exp(log_u1)

        R = y_bc * aLy_bc - m
        v_r = max(0.0, 1e-10 - R)
        R_eff = max(R, 1e-10)

        sech2 = 1.0 - np.tanh(u1)**2

        # BC2: A*sech2 / (Delta_0 * f1^2) = R
        # u1 def: u1 = (1-c) / (Delta_0 * f1)
        # dividing: A*sech2 / ((1-c)*f1) = R/u1
        # => (1-c)*f1 = A*sech2*u1/R  -- solve for c
        target = A * sech2 * u1 / R_eff

        c = self._solve_c(delta, target)
        root_violation = 0.0 if c is not None else 1.0
        if c is None:
            c = 0.95

        f1 = max(1.0 + delta * (1.0 - c), self._WIDTH_FLOOR)
        Delta_0 = (1.0 - c) / (u1 * f1)

        v_d_lo = max(0.0, self._D0_BOUNDS[0] - Delta_0)
        v_d_hi = max(0.0, Delta_0 - self._D0_BOUNDS[1])

        violation = np.array([v_r, root_violation, v_d_lo, v_d_hi], dtype=float)

        b = y_bc - A * (1.0 - np.tanh(u1))

        return (A, Delta_0, delta, m, c, b), violation

    def _is_feasible_theta(
        self,
        p: Union[np.ndarray, Dict[str, float]],
        y_bc: float,
        aLy_bc: float,
    ) -> bool:
        phys = self._to_physical(p, y_bc, aLy_bc)
        return phys is not None

    def _fit_multistart_trf(
        self,
        residual_fn,
        y_bc: float,
        aLy_bc: float,
        p0: Optional[np.ndarray] = None,
        accept_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> np.ndarray:
        """Fit with 2 fixed multistart points and optional warm-start seed."""
        bounds = self._global_bounds()
        bounds_lo = np.array([b[0] for b in bounds], dtype=float)
        bounds_hi = np.array([b[1] for b in bounds], dtype=float)

        # Two fixed starting points.
        fixed_starts = [
            np.array([np.log(0.001), np.log(1e-2), 0.0, 1.0], dtype=float),
            np.array([np.log(1.0), np.log(0.05), 0.0, 1.0], dtype=float),
        ]

        # Optional warm-start from cached/interpolated theta.
        candidates = []
        if p0 is not None:
            p0_arr = np.asarray(p0, dtype=float).reshape(-1)[:4]
            p0_arr = np.minimum(np.maximum(p0_arr, bounds_lo), bounds_hi)
            candidates.append(p0_arr)
        candidates.extend(fixed_starts)

        best_result = None
        best_theta = None

        for theta0 in candidates:
            if not self._is_feasible_theta(theta0, y_bc, aLy_bc):
                continue
            try:
                result = least_squares(
                    residual_fn,
                    theta0,
                    method='trf',
                    bounds=(bounds_lo, bounds_hi),
                    ftol=self.fit_trf_ftol,
                    gtol=self.fit_trf_gtol,
                    xtol=self.fit_trf_xtol,
                    diff_step=self.fit_trf_diff_step,
                    x_scale='jac',
                    max_nfev=self.fit_max_nfev,
                )
            except Exception:
                continue

            if best_result is None or result.cost < best_result.cost:
                best_result = result
                best_theta = np.asarray(result.x, dtype=float)
            if accept_fn is not None and result.success:
                cand_theta = np.asarray(result.x, dtype=float)
                if accept_fn(cand_theta):
                    return cand_theta

        if best_theta is not None:
            return best_theta

        return np.array([np.log(0.5), np.log(1.0), 0.0, 1.0], dtype=float)

    def _global_bounds(self) -> List[Tuple[float, float]]:
        return [
            (float(np.log(self._A_BOUNDS[0])), float(np.log(self._A_BOUNDS[1]))),
            (float(np.log(self._U1_BOUNDS[0])), float(np.log(self._U1_BOUNDS[1]))),
            (float(self._DELTA_BOUNDS[0]), float(self._DELTA_BOUNDS[1])),
            (0.0, 10.0),
        ]

    def _fit_mtanh_to_knots(
        self,
        x_k: np.ndarray,
        y_k: np.ndarray,
        y_bc: float,
        aLy_bc: float,
        p0: Optional[np.ndarray] = None,
        _input_vec: Optional[np.ndarray] = None,
        _prof: str = '',
    ) -> Dict[str, float]:
        x_k = np.asarray(x_k, dtype=float)
        y_k = np.asarray(y_k, dtype=float)
        y_scale = np.maximum(np.abs(y_k), 1e-3)

        def _residuals(p: np.ndarray) -> np.ndarray:
            phys, violation = self._to_physical_projected(p, y_bc, aLy_bc)
            A, D0, delta, m, c, b = phys
            res_data = (self._y_mtanh(x_k, A, D0, delta, m, c, b) - y_k) / y_scale
            # Keep smooth penalties: prior flat/hard-wall penalties led to solver pathologies.
            penalty = np.sqrt(self.fit_penalty_weight) * violation
            return np.concatenate([res_data, penalty])

        def _accept(p: np.ndarray) -> bool:
            phys = self._to_physical(p, y_bc, aLy_bc)
            if phys is None:
                return False
            A, D0, delta, m, c, b = phys
            model = self._y_mtanh(x_k, A, D0, delta, m, c, b)
            rel = np.abs(model - y_k) / np.maximum(np.abs(y_k), 1e-3)
            return bool(np.max(rel) <= self.fit_max_rel_error)

        popt = self._fit_multistart_trf(_residuals, y_bc, aLy_bc, p0=p0, accept_fn=_accept)
        return {
            'log_A': float(popt[0]),
            'log_u1': float(popt[1]),
            'delta': float(popt[2]),
            'm': float(popt[3]),
        }

    def _fit_mtanh_to_alys(
        self,
        x_k: np.ndarray,
        aLy_k: np.ndarray,
        y_bc: float,
        aLy_bc: float,
        p0: Optional[np.ndarray] = None,
        _input_vec: Optional[np.ndarray] = None,
        _prof: str = '',
    ) -> Dict[str, float]:
        x_fit = np.append(np.asarray(x_k, dtype=float), 1.0)
        aLy_fit = np.append(np.asarray(aLy_k, dtype=float), float(aLy_bc))
        scale = np.maximum(aLy_fit, 1e-3)

        def _residuals(p: np.ndarray) -> np.ndarray:
            phys, violation = self._to_physical_projected(p, y_bc, aLy_bc)
            A, D0, delta, m, c, b = phys
            y = self._y_mtanh(x_fit, A, D0, delta, m, c, b)
            dydx = self._dydx_mtanh(x_fit, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)
            aLy_model = -dydx / y_safe
            res_data = (aLy_model - aLy_fit) / scale
            penalty = np.sqrt(self.fit_penalty_weight) * violation
            return np.concatenate([res_data, penalty])

        def _accept(p: np.ndarray) -> bool:
            phys = self._to_physical(p, y_bc, aLy_bc)
            if phys is None:
                return False
            A, D0, delta, m, c, b = phys
            y = self._y_mtanh(x_fit, A, D0, delta, m, c, b)
            dydx = self._dydx_mtanh(x_fit, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)
            model = -dydx / y_safe
            rel = np.abs(model - aLy_fit) / np.maximum(np.abs(aLy_fit), 1e-3)
            return bool(np.max(rel) <= self.fit_max_rel_error)

        popt = self._fit_multistart_trf(_residuals, y_bc, aLy_bc, p0=p0, accept_fn=_accept)
        return {
            'log_A': float(popt[0]),
            'log_u1': float(popt[1]),
            'delta': float(popt[2]),
            'm': float(popt[3]),
        }

    def _verify_resolve(
        self,
        vec: np.ndarray,
        phys: Tuple[float, float, float, float, float, float],
        y_bc: float,
        aLy_bc: float,
    ) -> bool:
        """Check that physical params reproduce the input knot values within fit_max_rel_error."""
        A, D0, delta, m, c, b = phys
        n = len(self.knots)
        if self.defined_on == 'aLy':
            x_chk = np.append(self.knots, 1.0)
            target = np.append(vec[:n], aLy_bc)
            scale = np.maximum(np.abs(target), 1e-3)
            y = self._y_mtanh(x_chk, A, D0, delta, m, c, b)
            dydx = self._dydx_mtanh(x_chk, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)
            model = -dydx / y_safe
        else:
            y_k = np.array([y_bc + float(np.sum(vec[i:])) for i in range(n)], dtype=float)
            x_chk = np.append(self.knots, 1.0)
            target = np.append(y_k, y_bc)
            scale = np.maximum(np.abs(target), 1e-3)
            model = self._y_mtanh(x_chk, A, D0, delta, m, c, b)
        return bool(np.max(np.abs(model - target) / scale) <= self.fit_max_rel_error)

    def _resolve(
        self,
        prof: str,
        prof_params: Union[Dict[str, float], np.ndarray, Sequence[float]],
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        bc_y = self.get_nearest_bc(prof, 1.0)
        bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
        y_bc = float(bc_y['val']) if bc_y is not None else 1.0
        aLy_bc = float(bc_aLy['val']) if bc_aLy is not None else 1.0

        n = len(self.knots)
        if isinstance(prof_params, dict):
            vec = np.array([float(prof_params[name]) for name in self.param_names], dtype=float)
        else:
            vec = np.asarray(prof_params, dtype=float).reshape(-1)
            if vec.size < len(self.param_names):
                raise ValueError(f"Expected at least {len(self.param_names)} params for '{prof}', got {vec.size}")
            vec = vec[:n].astype(float, copy=False)

        # Cache lookup: return immediately if a prior fit is within tolerance.
        cache_key = np.concatenate([vec, [y_bc, aLy_bc]])
        p0 = self._last_theta_guess.get(prof)
        if self.fit_cache_enabled:
            entries = self._resolve_cache.get(prof, [])
            # 1. Exact hit: any entry within tolerance.
            for ck, cached_theta, cp in entries:
                denom = np.maximum(np.abs(ck), 1e-8)
                if np.max(np.abs(cache_key - ck) / denom) <= self.fit_cache_tol:
                    if self._verify_resolve(vec, cp, y_bc, aLy_bc):
                        return cp
                    p0 = cached_theta
                    break
            # 2. IDW interpolation over nearby entries in solver space,
            #    reprojected with current BCs. Uses same acceptance criterion as fitter.
            #    Distance is computed over knot-param dimensions only (first n entries);
            #    BCs are excluded because they are correctly reprojected via _to_physical.
            if entries:
                keys = np.array([ck for ck, _, _ in entries])
                knot_keys = keys[:, :n]
                denom = np.maximum(np.abs(knot_keys), 1e-8)
                rel_dists = np.max(np.abs(vec - knot_keys) / denom, axis=1)
                nearby = np.where(rel_dists <= self.fit_cache_tol * 5)[0]
                if nearby.size > 0:
                    w = 1.0 / np.maximum(rel_dists[nearby], 1e-10)
                    w /= w.sum()
                    thetas = np.array([entries[i][1] for i in nearby])
                    interp_theta = w @ thetas
                    interp_phys = self._to_physical(interp_theta, y_bc, aLy_bc)
                    if interp_phys is not None and self._verify_resolve(vec, interp_phys, y_bc, aLy_bc):
                        # Verified: add as a new exact entry and return.
                        if len(entries) >= self.fit_cache_max_size:
                            entries.pop(0)
                        entries.append((cache_key, interp_theta, interp_phys))
                        return interp_phys
                    # Not good enough: use interpolated theta as warm start for the fitter.
                    p0 = interp_theta

        if self.defined_on == 'dy':
            y_k = np.array([y_bc + float(np.sum(vec[i:])) for i in range(n)], dtype=float)
            x_fit = np.append(self.knots, 1.0)
            y_fit = np.append(y_k, y_bc)
            fit = self._fit_mtanh_to_knots(x_fit, y_fit, y_bc, aLy_bc,
                                           p0=p0)
        else:
            fit = self._fit_mtanh_to_alys(self.knots, vec, y_bc, aLy_bc,
                                          p0=p0)

        seed = np.array([fit['log_A'], fit['log_u1'], fit['delta'], fit['m']], dtype=float)
        phys = self._to_physical(seed, y_bc, aLy_bc)
        if phys is None or not self._verify_resolve(vec, phys, y_bc, aLy_bc):
            return None

        self._last_theta_guess[prof] = seed

        if self.fit_cache_enabled and phys is not None:
            entries = self._resolve_cache.setdefault(prof, [])
            if len(entries) >= self.fit_cache_max_size:
                entries.pop(0)
            entries.append((cache_key, seed, phys))

        return phys

    def parameterize(
        self, state, bc_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        self.build_bcs(bc_dict)
        params: Dict[str, Dict[str, float]] = {}
        params_std: Dict[str, Dict[str, float]] = {}

        x_data = np.asarray(getattr(state, 'roa')).flatten()

        for prof in self.predicted_profiles:
            y_raw = np.asarray(getattr(state, prof)).flatten()
            bc_y = self.get_nearest_bc(prof, 1.0)
            bc_aLy = self.get_nearest_bc(f'aL{prof}', 1.0)
            y_bc = float(bc_y['val']) if bc_y is not None else float(y_raw[-1]) if len(y_raw) else 1.0
            aLy_bc = float(bc_aLy['val']) if bc_aLy is not None else 1.0

            p0_default = np.array([np.log(0.5), np.log(1.0), 0.0, 1.0], dtype=float)
            y_scale = np.maximum(y_raw, 1e-6)

            def _residual_joint(p):
                phys, violation = self._to_physical_projected(p, y_bc, aLy_bc)
                A, D0, delta, m, c, b = phys
                y_model = self._y_mtanh(x_data, A, D0, delta, m, c, b)
                res_data = (y_model - y_raw) / y_scale
                penalty = np.sqrt(self.fit_penalty_weight) * violation
                return np.concatenate([res_data, penalty])

            popt = self._fit_multistart_trf(_residual_joint, y_bc, aLy_bc)

            phys = self._to_physical(popt, y_bc, aLy_bc)
            if phys is None:
                phys = self._to_physical(p0_default, y_bc, aLy_bc)
                if phys is None:
                    params[prof] = {name: 0.0 for name in self.param_names}
                    params_std[prof] = {name: 0.0 for name in self.param_names}
                    continue
            A, D0, delta, m, c, b = phys

            n = len(self.knots)
            if self.defined_on == 'dy':
                y_k = self._y_mtanh(self.knots, A, D0, delta, m, c, b)
                dy = np.empty(n, dtype=float)
                for i in range(n - 1):
                    dy[i] = float(y_k[i]) - float(y_k[i + 1])
                dy[n - 1] = float(y_k[n - 1]) - y_bc
                pdict = {f'dy{i}': float(dy[i]) for i in range(n)}
            else:
                dydx_k = self._dydx_mtanh(self.knots, A, D0, delta, m, c)
                y_k = self._y_mtanh(self.knots, A, D0, delta, m, c, b)
                y_safe = np.where(np.abs(y_k) < self._Y_FLOOR, self._Y_FLOOR, y_k)
                aLy_vec = np.clip(-dydx_k / y_safe, 0.0, None)
                pdict = {f'aLy{i}': float(aLy_vec[i]) for i in range(n)}

            params[prof] = pdict
            params_std[prof] = {k: abs(v) * self.sigma for k, v in pdict.items()}

        self.params = params
        self.params_std = params_std
        return params, params_std

    def _evaluate_once(
        self, batch_params: Dict[str, Any], x_1d: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """Resolve all profiles for a single (non-batched) set of params and x grid."""
        x_1d = np.asarray(x_1d, dtype=float)
        y_out: Dict[str, np.ndarray] = {}
        aLy_out: Dict[str, np.ndarray] = {}
        curv_out: Dict[str, np.ndarray] = {}

        for prof, pv in batch_params.items():
            phys = self._resolve(prof, pv)
            if phys is None:
                vec = (
                    np.array([float(pv[name]) for name in self.param_names], dtype=float)
                    if isinstance(pv, dict)
                    else np.asarray(pv, dtype=float).reshape(-1)[: len(self.knots)]
                )
                # print(
                #     f"[SplineMtanh] WARNING: no acceptable Mtanh fit for profile '{prof}' "
                #     f"(fit_max_rel_error={self.fit_max_rel_error:.3g}, "
                #     f"knot params={vec.tolist()}). Falling back to Akima spline."
                # )
                # Lazily construct an Akima Spline fallback with the same knots/defined_on.
                if not hasattr(self, '_spline_fallback') or self._spline_fallback is None:
                    fallback_options = {
                        'knots': self.knots.tolist(),
                        'defined_on': self.defined_on,
                        'spline_type': 'akima',
                        'predicted_profiles': self.predicted_profiles,
                        'include_zero_grad_on_axis': self.include_zero_grad_on_axis,
                        'sigma': self.sigma,
                    }
                    self._spline_fallback = Spline(fallback_options)
                self._spline_fallback.bc_dict = self.bc_dict
                single_params = {prof: pv}
                y_fb   = self._spline_fallback.get_y(single_params, x_1d)
                aLy_fb = self._spline_fallback.get_aLy(single_params, x_1d)
                curv_fb = self._spline_fallback.get_curvature(single_params, x_1d)
                y_out[prof]    = y_fb[prof]
                aLy_out[prof]  = aLy_fb[prof]
                curv_out[prof] = curv_fb[prof]
                continue

            A, D0, delta, m, c, b = phys
            y = np.clip(self._y_mtanh(x_1d, A, D0, delta, m, c, b), 0.0, None)
            dydx = self._dydx_mtanh(x_1d, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)

            y_out[prof] = y
            aLy_out[prof] = np.clip(-dydx / y_safe, 0.0, None)
            curv_out[prof] = self._d2ydx2_mtanh(x_1d, A, D0, delta, c)

        return y_out, aLy_out, curv_out, batch_params

    def update(self, params: Dict[str, np.ndarray], bc_dict: Dict[str, Any], x_eval: np.ndarray):
        """Evaluate SplineMtanh outputs with a single resolve per profile.

        The base class computes y, aLy, and curvature via separate method calls,
        which triggers repeated _resolve/_fit work for SplineMtanh. This
        override resolves once per profile (per batch item) and derives all
        outputs from that single resolved physical state.
        """
        self.bc_tensors = {}
        n_knots = len(self.knots)

        def _to_numpy(value: Any) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def _slice_params_for_batch(
            all_params: Dict[str, Any], batch_idx: int, batch_size: int
        ) -> Dict[str, Any]:
            sliced: Dict[str, Any] = {}
            for prof, prof_params in all_params.items():
                if isinstance(prof_params, dict):
                    prof_sliced: Dict[str, Any] = {}
                    for name, value in prof_params.items():
                        arr = _to_numpy(value)
                        if arr.ndim == 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = float(arr[batch_idx])
                        elif arr.ndim > 1 and arr.shape[0] == batch_size:
                            prof_sliced[name] = arr[batch_idx]
                        else:
                            prof_sliced[name] = value
                    sliced[prof] = prof_sliced
                else:
                    arr = _to_numpy(prof_params)
                    if arr.ndim > 1 and arr.shape[0] == batch_size:
                        sliced[prof] = arr[batch_idx]
                    else:
                        sliced[prof] = prof_params
            return sliced

        def _evaluate_once(batch_params: Dict[str, Any], x_1d: np.ndarray):
            return self._evaluate_once(batch_params, x_1d)

        x_arr = _to_numpy(x_eval)
        if x_arr.ndim <= 1:
            self.build_bcs(bc_dict)
            y, aLy, curv, _ = _evaluate_once(params, x_arr)
            self.y = y
            self.aLy = aLy
            self.curv = curv
            return y, aLy, curv

        batch_size = x_arr.shape[0]
        use_batched_bcs = self._is_batched_bc_input(bc_dict, batch_size)
        if use_batched_bcs:
            self.bc_tensors = bc_dict
        else:
            self.build_bcs(bc_dict)

        y_batches: Dict[str, List[np.ndarray]] = {}
        aLy_batches: Dict[str, List[np.ndarray]] = {}
        curv_batches: Dict[str, List[np.ndarray]] = {}

        for i in range(batch_size):
            if use_batched_bcs:
                self.build_bcs(self._slice_batched_bc_dict(bc_dict, i, batch_size))
            y_i, aLy_i, curv_i, _ = _evaluate_once(
                _slice_params_for_batch(params, i, batch_size),
                x_arr[i],
            )
            for prof, vals in y_i.items():
                y_batches.setdefault(prof, []).append(np.asarray(vals, dtype=float))
            for prof, vals in aLy_i.items():
                aLy_batches.setdefault(prof, []).append(np.asarray(vals, dtype=float))
            for prof, vals in curv_i.items():
                curv_batches.setdefault(prof, []).append(np.asarray(vals, dtype=float))

        y_out = {prof: np.stack(vals, axis=0) for prof, vals in y_batches.items()}
        aLy_out = {prof: np.stack(vals, axis=0) for prof, vals in aLy_batches.items()}
        curv_out = {prof: np.stack(vals, axis=0) for prof, vals in curv_batches.items()}

        self.y = y_out
        self.aLy = aLy_out
        self.curv = curv_out
        return y_out, aLy_out, curv_out

    def get_y(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            phys = self._resolve(prof, pv)
            if phys is None:
                out[prof] = self._penalty_vec(x_eval.size)
                continue
            A, D0, delta, m, c, b = phys
            out[prof] = np.clip(self._y_mtanh(x_eval, A, D0, delta, m, c, b), 0.0, None)
        self.y = out
        return out

    def get_aLy(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            phys = self._resolve(prof, pv)
            if phys is None:
                out[prof] = self._penalty_vec(x_eval.size)
                continue
            A, D0, delta, m, c, b = phys
            y = self._y_mtanh(x_eval, A, D0, delta, m, c, b)
            dydx = self._dydx_mtanh(x_eval, A, D0, delta, m, c)
            y_safe = np.where(np.abs(y) < self._Y_FLOOR, self._Y_FLOOR, y)
            out[prof] = np.clip(-dydx / y_safe, 0.0, None)
        self.aLy = out
        return out

    def get_curvature(
        self, params: Dict[str, Dict[str, float]], x_eval: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_eval = np.asarray(x_eval)
        out: Dict[str, np.ndarray] = {}
        for prof, pv in params.items():
            phys = self._resolve(prof, pv)
            if phys is None:
                out[prof] = self._penalty_vec(x_eval.size)
                continue
            A, D0, delta, m, c, b = phys
            out[prof] = self._d2ydx2_mtanh(x_eval, A, D0, delta, c)
        self.curv = out
        return out

# -------------------------
# Factory and registry
# -------------------------


PARAMETER_MODELS = {
    'spline': Spline,
    'mtanh': Mtanh,
    'spline_mtanh': SplineMtanh,
}


def create_parameter_model(config: Dict[str, Any]) -> ParameterBase:
    """Create a parameter model instance from config.

    Expected config format:
    {"type": "spline"|"mtanh"|"gaussian"|"log_spline"|"log_slope_spline", "kwargs": { ... model options ... }}
    """
    model_type = (config or {}).get('type', 'spline')
    kwargs = (config or {}).get('kwargs', {})
    cls = PARAMETER_MODELS.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown parameter model type: {model_type}")
    return cls(kwargs)


BCEntry = Dict[str, float]  # {'val': float, 'loc': float}

def _normalize_single_bc(val: Union[tuple, list, dict]) -> BCEntry:
    """Accept (value, loc) tuple/list or {'value':..., 'location':...}"""
    if isinstance(val, dict):
        if "val" in val:
            v = float(val["val"])
        else:
            v = float(val.get("value", 0.0))
        if "loc" in val:
            loc = float(val["loc"])
        else:
            loc = float(val.get("location", 1.0))
    elif isinstance(val, (tuple, list)) and len(val) == 2:
        v, loc = val
        v, loc = float(v), float(loc)
    else:
        raise ValueError("BC must be (value,location) or dict{'value','location'}")
    return {'val': v, 'loc': loc}