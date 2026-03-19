import torch
import datetime
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools.optimizers.multivariate_tools import mitim_jacobian
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def DVdistanceMetric(xT):
    yG = []
    xG = []
    for i in range(xT.shape[0] - 1):
        yA = []
        for j in range(xT[: i + 1, :].shape[0]):
            yA.append(np.abs((xT[i + 1, :] - xT[j, :]) / xT[j, :]) * 100.0)
        yA = np.array(yA)
        # Store the closest difference to an existing point per each dimension
        yG.append(yA.min(axis=0))

        xG.append(i + 1)

    yG = np.array(yG)

    # Calculate the maximum distance in a dimension
    yG_max = yG.max(axis=1) if len(yG) > 0 else yG

    return xG, yG_max


def checkSolutionIsWithinBounds(x, bounds, maxExtrapolation=[0.0, 0.0], clipper = 1E-6):
    mi = bounds[0, :]
    ma = bounds[1, :]

    # Hard limits
    maxb, minb = ma, mi

    # Allow extrapolation (added clip to avoid numerical issues of points very close to the boundary)
    minb = mi - np.max([maxExtrapolation[0],clipper]) * (ma - mi) 
    maxb = ma + np.max([maxExtrapolation[1],clipper]) * (ma - mi)
    insideBounds = (x <= maxb).all() and (x >= minb).all()

    return insideBounds


def _rand_points(combined_model, n_points):
    """Build a (n_points, n_dims) tensor of random points uniformly sampled within bounds."""
    
    bounds = combined_model.bounds
    dtype  = combined_model.train_X.dtype
    device = combined_model.train_X.device
    bt = torch.zeros(2, len(bounds), dtype=dtype, device=device)
    for i, key in enumerate(bounds):
        bt[0, i] = bounds[key][0]
        bt[1, i] = bounds[key][1]
    return bt[0] + (bt[1] - bt[0]) * torch.rand(n_points, bt.shape[-1], dtype=dtype, device=device)


def _detach_nonleaf_tensors(*roots):
    """
    Walk every root's attribute tree and replace non-leaf tensors in-place with
    their detached version.  Called after a grad-enabled forward pass to prevent
    deepcopy from failing on computation-graph nodes stored in model caches
    (e.g. tf1._cached_factor written by _batched_posterior).
    """
    visited = set()

    def _walk(obj):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, torch.Tensor):
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, torch.Tensor) and not v.is_leaf:
                    obj[k] = v.detach()
                else:
                    _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        else:
            d = getattr(obj, '__dict__', None)
            if d:
                for k, v in list(d.items()):
                    if isinstance(v, torch.Tensor) and not v.is_leaf:
                        d[k] = v.detach()
                    else:
                        _walk(v)

    for root in roots:
        _walk(root)


def _jacobian_mean(model, X, _also_clean=None):
    """
    Jacobian of the GP mean w.r.t. inputs, shape (n_points, n_outputs, n_inputs).
    Delegates to mitim_jacobian and cleans up non-leaf tensors afterward.
    """
    _, J = mitim_jacobian(lambda x: model.predict(x)[0], X)
    _detach_nonleaf_tensors(model, *(_also_clean or []))
    return J     # (n_pts, n_out, n_in)


def testInferenceTime(combined_model, n_points_list=[1000], additional_calls=None):
    """Time combined_model mean inference and Jacobian at n_points random points."""

    for n_points in n_points_list:
        
        print(f"\n[MITIM: GP performance] Testing inference time of evaluating {n_points} points...", typeMsg="i")

        X_rand = _rand_points(combined_model, n_points)
        n_dims = X_rand.shape[-1]

        # --- mean inference ---
        t0 = datetime.datetime.now()
        with torch.no_grad():
            mean,_,_,_ = combined_model.predict(X_rand)
        t_diff = IOtools.getTimeDifference(t0)
            
        n_gps = mean.shape[-1]
        n_train = combined_model.train_X.shape[0]
        print(
            f"\t- Mean inference ({n_dims}D, {n_gps} GPs, {n_train} training pts, {n_points} inference pts): "
            f"{t_diff}", typeMsg="i"
        )

        # --- Jacobian ---
        t0 = datetime.datetime.now()
        _jacobian_mean(combined_model, X_rand)
        t_diff = IOtools.getTimeDifference(t0)
        print(
            f"\t- Jacobian of mean ({n_dims}D, {n_gps} GPs, {n_train} training pts, {n_points} inference pts): "
            f"{t_diff}", typeMsg="i"
        )
        
        # --- additional call if requested ---
        if additional_calls is not None:
            for name, func in additional_calls.items():
                t0 = datetime.datetime.now()
                func(X_rand)
                t_diff = IOtools.getTimeDifference(t0, niceText=False)*1000
                print(
                    f"\t- {name}: {n_dims}D, {n_points} inference pts): "
                    f"{t_diff} ms", typeMsg="i"
                )
            # Return the time in ms of the last additional call for potential use in tests
            return t_diff
            

def testBatchAccuracy(combined_model, individual_models, n_points=1000, n_points_jac=5, thr_percent=0.1):
    """
    Verify that the batched combined_model gives the same predictions as the
    individual models evaluated sequentially.

    Mean and std are checked at n_points random points.
    Jacobian is checked at n_points_jac points (kept small: the reference requires
    n_out backward passes through all sequential individual models per point).
    """

    print(f"[MITIM: GP batching] Testing accuracy of combined_model predictions against individual models...", typeMsg="i")

    x     = _rand_points(combined_model, n_points)
    x_jac = _rand_points(combined_model, n_points_jac)

    # --- mean and std ---
    with torch.no_grad():
        y_batch, upper_batch, lower_batch, _ = combined_model.predict(x)
        indiv_preds = [m.predict(x) for m in individual_models]
        y_indiv     = torch.cat([p[0] for p in indiv_preds], dim=1)
        # upper/lower are ±2*std, so std = (upper - lower) / 4
        std_batch = ((upper_batch - lower_batch) / 4).detach()
        std_indiv = torch.cat([(p[1] - p[2]) / 4 for p in indiv_preds], dim=1).detach()

    y_batch = y_batch.detach()
    y_indiv = y_indiv.detach()

    mask_mean    = y_indiv.abs() > 1e-10
    err_mean     = torch.where(mask_mean, (y_batch - y_indiv).abs() / y_indiv.abs() * 100,
                               torch.zeros_like(y_batch))
    max_err_mean = err_mean.max().item()

    mask_std    = std_indiv.abs() > 1e-10
    err_std     = torch.where(mask_std, (std_batch - std_indiv).abs() / std_indiv.abs() * 100,
                              torch.zeros_like(std_batch))
    max_err_std = err_std.max().item()

    # --- Jacobian of mean (few points: reference is expensive) ---
    class _IndivWrapper:
        """Thin wrapper so _jacobian_mean can call predict() on the individual models."""
        def predict(self, X):
            mean = torch.cat([m.predict(X)[0] for m in individual_models], dim=1)
            return mean, None, None, None

    J_batch = _jacobian_mean(combined_model, x_jac)                              # (n_pts_jac, n_out, n_in)
    J_indiv = _jacobian_mean(_IndivWrapper(), x_jac, _also_clean=individual_models)  # (n_pts_jac, n_out, n_in)

    mask_jac    = J_indiv.abs() > 1e-10
    err_jac     = torch.where(mask_jac, (J_batch - J_indiv).abs() / J_indiv.abs() * 100,
                              torch.zeros_like(J_batch))
    max_err_jac = err_jac.max().item()

    passed = (max_err_mean <= thr_percent) and (max_err_std <= thr_percent) and (max_err_jac <= thr_percent)

    if not passed:
        print(
            f"\t- Accuracy check FAILED (threshold {thr_percent}%) "
            f"— batched and sequential predictions disagree\n"
            f"\t  mean     max relative error = {max_err_mean:.2e}%  ({n_points} pts)\n"
            f"\t  std      max relative error = {max_err_std:.2e}%  ({n_points} pts)\n"
            f"\t  Jacobian max relative error = {max_err_jac:.2e}%  ({n_points_jac} pts)",
            typeMsg="w",
        )
    else:
        print(
            f"\t- Accuracy check passed: "
            f"mean = {max_err_mean:.2e}% ({n_points} pts), "
            f"std = {max_err_std:.2e}% ({n_points} pts), "
            f"Jacobian = {max_err_jac:.2e}% ({n_points_jac} pts)",
            typeMsg="i",
        )

def isOutlier(y0, y, stds_outside=5, stds_outside_checker=1):
    mean = y.mean()
    stds = y.std()

    yu = mean + stds_outside * stds
    yl = mean - stds_outside * stds

    outlier = ((y0 < yl) or (y0 > yu)) and (y.shape[0] > stds_outside_checker)

    return outlier


def lookForTrouble(x, y_res, z_res, evaluators, stepSettings, elimintateTroubles=False):
    """
    This is to check that each optimization workflow has calculated correctly the acquisition for each member.
    GPYtorch is not robust enough, check here just in case
    """

    y_res_joint = evaluators["acq_function"](x.unsqueeze(1)).detach()

    y_res_single = torch.cat(
        [evaluators["acq_function"](x[i].unsqueeze(0).unsqueeze(1)).detach() for i in range(x.shape[0])],
        axis=0,
    ).to(x)

    perMax1, trouble1, indeces1 = checkSame(
        y_res, y_res_joint, z=z_res, labels=["OPTIMIZATION", "JOINT"]
    )
    perMax2, trouble2, indeces2 = checkSame(
        y_res, y_res_single, z=z_res, labels=["OPTIMIZATION", "SINGLE"]
    )
    perMax3, trouble3, indeces3 = checkSame(
        y_res_joint, y_res_single, z=z_res, labels=["JOINT", "SINGLE"]
    )

    if len(indeces2) > 0:
        numBad = len(indeces2)
        if elimintateTroubles:
            print(
                "\n\t- It has been requested to eliminate troubled points in positions:",
                indeces2,
            )
            x = np.delete(x, indeces2, axis=0)
            y_res = np.delete(y_res, indeces2, axis=0)
            z_res = np.delete(z_res, indeces2, axis=0)
        else:
            print(
                "\n\t- No action taken, but found troubled points in positions:",
                indeces2,
            )
    else:
        numBad = 0

    return x, y_res, z_res, numBad


def checkSame(
    y1o, y2o, z=None, thresholdTrigger=0.5, absoluteTrigger=1e-3, labels=["", ""]
):
    print(
        f"\t\t- Checking evaluation quality between {labels[0]} and {labels[1]}",
    )

    try:
        y1 = y1o.detach().cpu().numpy()
        y2 = y2o.detach().cpu().numpy()
    except:
        y1 = y1o
        y2 = y2o

    percents, absloutes, absloutes2 = [], [], []
    for i in range(y1.shape[0]):
        per = (np.abs(y1[i] - y2[i]) / y1[i]) * 100.0
        percents.append(per)
        absloutes.append(np.abs(y1[i]))
        absloutes2.append(np.abs(y2[i]))
    percents, absloutes, absloutes2 = (
        np.array(percents),
        np.array(absloutes),
        np.array(absloutes2),
    )

    aError = np.where((percents > thresholdTrigger))[0]
    a = np.where((percents > thresholdTrigger) & (absloutes2 > absoluteTrigger))[0]

    if len(a) == 0:
        if len(aError) == 0:
            print(
                "\t\t\t~ Evaluators provided error in all individuals less than {0:.1e}% (< {1:.1f}%)".format(
                    percents.max(), thresholdTrigger
                ),
            )
        else:
            print(
                "\t\t\t~ Evaluators provided error in all individuals of {0:.1e}% (> {1:.1f}%), but the absolute value is very low".format(
                    percents.max(), thresholdTrigger
                ),
            )
        trouble = False
    else:
        trouble = True
        print(
            "\t\t\t~ Evaluators provided error more than {0:.1f}% in following individuals:".format(
                thresholdTrigger
            ),
            typeMsg="w",
        )
        for i in a:
            if z is not None:
                extratxt = f". Evaluated by {identifyType(z[i])}"
            else:
                extratxt = ""
            try:
                print(
                    "\t\t\t #{0} with an evaluated value of {1:.1e} and error percent of {2:.1f}% (absolute evaluator: {3:.1e}){4}".format(
                        i, absloutes[i], percents[i], absloutes2[i], extratxt
                    ),
                    typeMsg="w",
                )
            except:
                print(
                    "\t\t\t #{0} with an evaluated value of {1} and error percent of {2}% (absolute evaluator: {3}){4}".format(
                        i, absloutes[i], percents[i], absloutes2[i], extratxt
                    ),
                    typeMsg="w",
                )

    return percents.max(), trouble, a


def summaryTypes(z_opt):
    types = ""
    for i in [0, 1, 2, 3, 4, 5, 6]:
        types += f"{(z_opt == i).sum()} from {identifyType(i)}, "

    return types[:-2]


def identifyType(z):
    if z == 0.0:
        method = "Previous Iteration"
    elif z == 1.0:
        method = "Trained"
    elif z == 2.0:
        method = "Random"
    elif z == 3.0:
        method = "BOTORCH"
    elif z == 4.0:
        method = "GA"
    elif z == 5.0:
        method = "ROOT"
    elif z == 6.0:
        method = "SR"

    return method
