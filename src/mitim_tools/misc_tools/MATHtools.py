import pdb
import copy
import scipy
import torch
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline
from scipy.integrate import romb
from mitim_tools.misc_tools import IOtools
from IPython import embed


def is_inBetween(x, x1, x2):
    [d1, d2] = calculateDistance(x, [x1, x2])
    dT = calculateDistance(x1, [x2])[0]

    if np.abs(dT - (d1 + d2)) < 1e-5:
        return True
    else:
        return False


def calculateDistanceAlong(x, limRZ, iS=3):
    print("Finding limiting points")
    prev_index = np.zeros(x.shape[0])
    for k in range(x.shape[0]):
        # Find limiting points
        for j in range(len(limRZ[iS:]) - 1):
            if is_inBetween(x[k], limRZ[iS + j], limRZ[iS + j + 1]):
                prev_index[k] = j + iS
                break

    print("Calculate distance along")
    d = np.zeros(x.shape[0])
    for k in range(x.shape[0]):
        for i in range(int(prev_index[k] - iS)):
            d[k] += calculateDistance(limRZ[iS + i + 1], [limRZ[iS + i]])[0]
        d[k] += calculateDistance(x[k], [limRZ[int(prev_index[k])]])[0]

    return d


def surface_integral(x, y, z_xy):
    """
    Taken the way freegs does it to integrate Jtor for Ip
    """

    dx = x[1, 0] - x[0, 0]
    dy = y[0, 1] - y[0, 0]

    I = romb(romb(z_xy)) * dx * dy

    return I


def MITIM_rosen(x, y, a=1, b=100):
    """
    https://en.wikipedia.org/wiki/Rosenbrock_function
    Global minimum with f=0 is at (a,a**2)
    """
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenGrid(x=np.linspace(-2, 2, 1000), ax=None):
    # HERE, otherwise it will require that it will require that I have botorch even for small MITIM use
    from mitim_tools.opt_tools.utils import BOgraphics

    y = x
    x0, y0 = np.meshgrid(x, y)

    z = MITIM_rosen(x0, y0)

    if ax is not None:
        BOgraphics.plot2D(ax, x0, y0, z, flevels=len(x), levels=[1e-2, 1e-1])

    return z


def create2Dmesh(x, y):
    x0, y0 = np.meshgrid(x, y)
    x0, y0 = np.atleast_2d(np.hstack(x0)), np.atleast_2d(np.hstack(y0))

    X = np.transpose(np.append(x0, y0, axis=0))

    return X


def interpolateM(x, xp, yp):
    f = scipy.interpolate.interp1d(xp, yp, kind="linear")

    return f(x)


def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike


def orderArray(arr, base=None):
    """
    Ordering from lowest (position #0) to highest (last position)
    """

    if base is None:
        base = arr

    sortedList = sorted(zip(base, arr))
    return np.array([a for _, a in sortedList])


def arePointsEqual(x1, x2):
    isEqual = True
    for i in range(x1.shape[0]):
        isEqual = isEqual and (x1[i] == x2[i])

    return isEqual


def simple_deriv(x, y):
    z = torch.zeros(len(y))

    for i in range(len(z)):
        if i == 0:
            z[i] = (y[1] - y[0]) / (x[1] - x[0])
        elif i == len(z) - 1:
            z[i] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            z[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

    return z


def deriv_fit(x, y, xbase, order=2, debug=False):
    x_grid, y_grid, fit_params = poly_fit(x, y, order=order)

    der = deriv(x_grid, y_grid)
    der_x = der[np.argmin(np.abs(x_grid - xbase))]

    if debug:
        fig, ax = plt.subplots()
        ax.plot(x, y, "-o", c="b")
        ax.plot(x_grid, y_grid, lw=2, c="r")
        plt.show()

    return der_x, der, x_grid, y_grid, fit_params


def poly_fit(x, y, order=2, x_grid=None):
    fit_params = np.polyfit(x, y, order)
    if x_grid is None:
        x_grid = np.linspace(0, x[-1], 1000)
    y_grid = np.zeros(len(x_grid))
    for i in range(len(fit_params)):
        y_grid += fit_params[i] * x_grid ** (order - i)

    return x_grid, y_grid, fit_params


def smoothThroughSawtooth(t, varorig, tsaws, howmanybefore, Delta_tBefore=0):
    taux = 0
    tnew = []
    varnew = []

    for it in range(len(t)):
        if tsaws[it] > taux:
            # If just taking the value before sawtooth
            if Delta_tBefore == 0:
                tnew.append(t[it - howmanybefore])
                varnew.append(varorig[it - howmanybefore])
            # If averaging from tsaw[howmanybefore]-Delta_tBefore to tsaw[howmanybefore]
            else:
                itS = np.argmin(np.abs(t - (t[it - howmanybefore] - Delta_tBefore)))
                tnew.append(np.mean(t[itS : it - howmanybefore]))
                varnew.append(np.mean(varorig[itS : it - howmanybefore]))
            taux = tsaws[it]

    # Remove first one because it is just the initial point, not sawtooth?
    varnew = np.array(varnew[1:])
    tnew = np.array(tnew[1:])

    return varnew, tnew


def deriv(x, y, array=True):
    """
    This function returns the derivative of the 2nd order lagrange interpolating polynomial of y(x)
    from OMFIT, and also consistent with bound_deriv(df,f,r,n) in expro_util
    """

    if array:
        x, y = np.array(x), np.array(y)

    def dlip(ra, r, f):
        """dlip - derivative of lagrange interpolating polynomial"""
        r1, r2, r3 = r
        f1, f2, f3 = f
        return (
            ((ra - r1) + (ra - r2)) / (r3 - r1) / (r3 - r2) * f3
            + ((ra - r1) + (ra - r3)) / (r2 - r1) / (r2 - r3) * f2
            + ((ra - r2) + (ra - r3)) / (r1 - r2) / (r1 - r3) * f1
        )

    fin = (
        [dlip(x[0], x[0:3], y[0:3])]
        + list(dlip(x[1:-1], [x[0:-2], x[1:-1], x[2:]], [y[0:-2], y[1:-1], y[2:]]))
        + [dlip(x[-1], x[-3:], y[-3:])]
    )

    if array:
        return np.array(fin)
    else:
        return torch.Tensor(fin).to(x)  # torch.from_numpy(np.array(fin)).to(x)


def applyNiche(x, y=None, tol=1e-3):
    """
    tol is the abolute distance threshold, so bounding x between 0 and 1 prior to this is a good idea
    """
    x_orig = copy.deepcopy(x)

    yN = copy.deepcopy(y)

    for i in range(10000):
        # Find those positions that are too close to present one
        if i > len(x) - 1:
            break
        d = calculateDistance(x[i], x)
        ident = np.arange(0, len(d))[d < tol]
        ident = ident[ident != i]
        x = np.delete(x, ident, 0)
        if y is not None:
            yN = np.delete(yN, ident, 0)

    if x.shape[0] < x_orig.shape[0]:
        print(f"\t\t\t\t- Niche correction of {tol} has been applied so {x_orig.shape[0]-x.shape[0]} members have been removed, remaining {x.shape[0]}")

    return x, yN


def calculateDistance(xi, x):
    distance = np.zeros(len(x))
    for i in range(len(x)):
        distance[i] = np.linalg.norm(np.array(x[i]) - np.array(xi))

    return distance


def integrate(x, deriv):
    return np.append([0], scipy.integrate.cumtrapz(deriv, x=x))


def integrate_definite(x, y, rangex=None):
    if len(x) > 1:
        # if rangex is None:  return integrate(x,y)[-1]
        # else:               return integrate(x,y)[np.argmin(np.abs(x-rangex[1]))] - integrate(x,y)[np.argmin(np.abs(x-rangex[0]))]
        return np.trapz(y, x=x)
    else:
        return 0


def integrateQuadPoly(r, s):
    """
    (batch,dim)

    Computes int(s*dr), so if s is s*dV/dr, then int(s*dV), which is the full integral

    From tgyro_volume_int.f90
    r - minor radius
    s - s*volp

    (Modified to avoid if statements and for loops)

    """

    if isinstance(s, torch.Tensor):
        p = torch.zeros((r.shape[0], r.shape[1])).to(r)
    else:
        p = np.zeros((r.shape[0], r.shape[1]))

    # First point

    x1, x2, x3 = r[..., 0], r[..., 1], r[..., 2]
    f1, f2, f3 = s[..., 0], s[..., 1], s[..., 2]

    p[..., 1] = (x2 - x1) * (
        (3 * x3 - x2 - 2 * x1) * f1 / 6 / (x3 - x1)
        + (3 * x3 - 2 * x2 - x1) * f2 / 6 / (x3 - x2)
        - (x2 - x1) ** 2 * f3 / 6 / (x3 - x1) / (x3 - x2)
    )

    # Next points
    x1, x2, x3 = r[..., :-2], r[..., 1:-1], r[..., 2:]
    f1, f2, f3 = s[..., :-2], s[..., 1:-1], s[..., 2:]

    p[..., 2:] = (
        (x3 - x2)
        / (x3 - x1)
        / 6
        * (
            (2 * x3 + x2 - 3 * x1) * f3
            + (x3 + 2 * x2 - 3 * x1) * f2 * (x3 - x1) / (x2 - x1)
            - (x3 - x2) ** 2 * f1 / (x2 - x1)
        )
    )

    if isinstance(p, torch.Tensor):
        return torch.cumsum(p, 1)
    else:
        return np.cumsum(p, 1)


def extrapolate(x, xp, yp, order=3):
    s = InterpolatedUnivariateSpline(xp, yp, k=order)

    return s(x)


def extrapolateCubicSpline(x, xp, yp):
    s = CubicSpline(xp, yp)  # ,bc_type='clamped')

    return s(x)


def circle(Raxis, num=100):
    theta = np.linspace(0, 2 * np.pi, int(num))
    R, Z = np.zeros(num), np.zeros(num)
    for i in range(len(theta)):
        R[i], Z[i] = Raxis * np.cos(theta[i]), Raxis * np.sin(theta[i])

    return R, Z


def HighDimSearch(vectorsLUT, valuesLUT, valuesEval, method="linear"):
    from scipy.interpolate import griddata

    val = []
    for i in range(len(vectorsLUT)):
        if valuesEval[i] < vectorsLUT[i].min():
            print(f"\n\n\t--> PROBLEM: value {i} < database minimum")
            val.append(vectorsLUT[i].min())
        elif valuesEval[i] > vectorsLUT[i].max():
            print(f"\n\n\t--> PROBLEM: value {i} > database maximum")
            val.append(vectorsLUT[i].max())
        else:
            val.append(valuesEval[i])
    valuesEval = tuple(val)

    val = griddata(vectorsLUT, valuesLUT, valuesEval, method=method)

    return val


def sigmoid_MITIM(xo, xbo, h=1e3):
    """
    xo is the points to evaluate, (batch, dim)
    xbo are the bounds (2,dim)
    h is strength of the sigmoid (0...1000). 1000 is basically a box

    returned is in (batch,dim), if 0 -> penalized
    """

    a = h / (xbo[1, :] - xbo[0, :])

    s_lower = 1 / (1 + torch.exp(-a * (xo - xbo[0, :])))
    s_upper = 1 / (1 + torch.exp(a * (xo - xbo[1, :])))

    s = s_lower * s_upper

    return s

def sigmoidPenalty(x, x_unity=[1, 10], h_transition=0.5):
    # h_transition is the width for 0.5

    [p1, p2] = x_unity
    h = h_transition

    return (
        1.0
        / (1 + np.exp(-10.0 / h * (x - p1 + h)))
        * 1.0
        / (1 + np.exp(10.0 / h * (x - p2 - h)))
    )


def intersect(RMC, YMC, chordRmaj):
    chordRmaj = np.interp(RMC, [chordRmaj, chordRmaj], [-1e3, 1e3])

    idx = np.argwhere(np.diff(np.sign(YMC - chordRmaj))).flatten()

    if len(idx) > 0:
        yi = YMC[idx]
    else:
        yi = None

    return yi


def divideZero(a, b):
    try:
        div = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    except:
        div = a / b

    return div


def downsampleCurve(x, y, nsamp=1000):
    gridx = np.linspace(0, len(x) - 1, len(x))
    gridx_new = np.linspace(0, len(x) - 1, int(nsamp))

    xnew = np.interp(gridx_new, gridx, x)
    ynew = np.interp(gridx_new, gridx, y)

    return xnew, ynew


def upsampleCurve(x, y, extra_factor=10, closeCurve=True, debug=False):
    """
    In between points, construct more points
    """

    extra_factor = int(extra_factor)

    xNew, yNew = [], []
    for i in range(len(x)):
        if i == len(x) - 1:
            if closeCurve:
                xnext, ynext = x[0], y[0]
            else:
                continue
        else:
            xnext, ynext = x[i + 1], y[i + 1]

        # Expand by linear interpoltion of x. It can go in both directions
        x0 = np.linspace(x[i], xnext, extra_factor)

        # If the next point is growing
        if xnext > x[i]:
            y0 = np.interp(x0, [x[i], xnext], [y[i], ynext])
        # If the next point is decreasing
        elif xnext < x[i]:
            y0 = np.flipud(np.interp(np.flipud(x0), [xnext, x[i]], [ynext, y[i]]))
        # If the next point is the same
        else:
            y0 = np.linspace(y[i], ynext, extra_factor)

        xNew.extend(x0)
        yNew.extend(y0)

    if debug:
        fig, ax = plt.subplots()
        ax.plot(xNew, yNew, "-o", c="r", lw=1.0, markersize=3)
        ax.plot(x, y, "-o", c="b", lw=0.5, markersize=3)
        plt.show()

    return np.array(xNew), np.array(yNew)


def smoothCurve(t, z, axis=0, Delta_t=20e-3):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    z = np.array(z)

    if len(z.shape) == 1:
        zn = lowess(
            z,
            t,
            is_sorted=True,
            frac=Delta_t / (t[-1] - t[0]),
            it=0,
            return_sorted=False,
        )
    elif axis == 0:
        zn = []
        for i in range(z.shape[1]):
            zn.append(
                lowess(
                    z[:, i],
                    t,
                    is_sorted=True,
                    frac=Delta_t / (t[-1] - t[0]),
                    it=0,
                    return_sorted=False,
                )
            )
        zn = np.transpose(np.array(zn))
    elif axis == 1:
        zn = []
        for i in range(z.shape[0]):
            zn.append(
                lowess(
                    z[i],
                    t,
                    is_sorted=True,
                    frac=Delta_t / (t[-1] - t[0]),
                    it=0,
                    return_sorted=False,
                )
            )
        zn = np.array(zn)

    return zn


def GaussianDistribution(rho, mu, sigma):
    return 1 / sigma / (2 * np.pi) ** 0.5 * np.exp(-((rho - mu) ** 2) / 2 / sigma**2)


def perturbativePulse(
    rho, t, Strength, mu_rho, v, mu_t, sigma_rho, sigma_t, skew_rho, skew_t
):
    PulseAll = []
    for it in range(len(t)):
        Pulse = np.zeros(len(rho))

        thresholdTime = mu_t

        # Changing the mean position in time
        mu_rhoNew = mu_rho - v * (t[it] - thresholdTime)

        # Apply skewed gaussians
        for ix in range(len(rho)):
            # In Space
            factorexp = -((mu_rhoNew - rho[ix]) ** 2) / (2.0 * sigma_rho**2.0)
            if factorexp > -20:
                ExpFactorX = np.exp(factorexp)
            else:
                ExpFactorX = 0.0
            factorerf = -skew_rho * (rho[ix] - mu_rhoNew) / (np.sqrt(2.0) * sigma_rho)
            if factorerf > 5:
                ErfcSkewX = 0.0
            else:
                ErfcSkewX = scipy.special.erfc(factorerf)

            # In Time
            factorexp = -((mu_t - t[it]) ** 2) / (2.0 * sigma_t**2.0)
            if factorexp > -20:
                ExpFactorT = np.exp(factorexp)
            else:
                ExpFactorT = 0.0
            factorerf = -skew_t * (t[it] - mu_t) / (np.sqrt(2.0) * sigma_t)
            if factorerf > 5:
                ErfcSkewT = 0.0
            else:
                ErfcSkewT = scipy.special.erfc(factorerf)

            try:
                TimeSpaceFactor = ExpFactorX * ExpFactorT * ErfcSkewX * ErfcSkewT
            except:
                embed()

            Pulse[ix] = Strength * TimeSpaceFactor

        PulseAll.append(Pulse)

    return np.array(PulseAll)


def fitCoreFunction(ne_orig, x, ne0, netop, ix, coeff=3, miny=None):
    ne = copy.deepcopy(ne_orig)

    if miny is not None:
        ne = ne * (miny / netop)
        netop = netop * (miny / netop)

    for i in range(len(x[:ix])):
        a = (ne0 - netop) / (1 - (1 - x[ix] ** 2) ** coeff)
        b = ne0 - a

        ne[i] = a * (1 - x[i] ** 2) ** coeff + b

    return ne


def smoothProfile_functionalForm(x, y, posx=0.95, coeff=3, maxy=None, miny=None):
    # This function smoothes the core profile to avoid discontinuities
    # It conserves pedestal value (ix position), core value (0 position)

    ix = np.argmin(np.abs(x - posx))

    if maxy is None:
        maxy = y[0]

    ynew = fitCoreFunction(y, x, maxy, y[ix], ix, coeff=coeff, miny=miny)

    return x, ynew


def profileMARS(peaking, average, rho=np.linspace(0, 1, 100)):
    return average * peaking * ((1 - rho**2) ** (peaking - 1)), rho


def profileMARS_MITIM(peaking, average, lambda_correction=2, rho=np.linspace(0, 1, 100)):
    c = lambda_correction  # TendTav/(1-TendTav)

    peaking = (c + 1) * peaking - c
    average = average / (c + 1)

    return average * (peaking * (1 - rho**2) ** (peaking - 1) + c), rho


def smoothProfile_continuity(x, z, posx=0.9):
    ix = np.argmin(np.abs(x - posx))

    pedestalx = x[ix:]
    pedestalz = z[ix:]

    corex = x[:ix]
    corez = z[:ix]

    corex, corez = BsplineFit(corex, corez)

    x = np.append(corex, pedestalx)
    z = np.append(corez, pedestalz)

    return x, z


def BsplineFit(xorig, yorig, howmanyPoints=20, plotDebug=False):
    x = xorig[:: int(len(xorig) / howmanyPoints)]
    y = yorig[:: int(len(yorig) / howmanyPoints)]

    t, c, k = interpolate.splrep(x, y, s=0, k=4)

    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    xnew = copy.deepcopy(xorig)
    ynew = spline(xnew)

    if plotDebug:
        fig, ax = plt.subplots()
        ax.plot(xorig, yorig, c="b")
        ax.plot(xnew, ynew, c="r")
        filef = IOtools.expandPath("~/scratch/smooth.svg")
        fig.savefig(filef)
        plt.close("all")
        pdb.set_trace()

    return xnew, ynew


def interp2D(x, y, xp, yp, fp, kind="cubic"):
    f = interpolate.interp2d(xp, yp, fp, kind=kind)

    return f(x, y)


def drawContours(Rold, Yold, Zold, resol=5e3, psiN_boundary=0.99999):
    # ~~~~~~~ Fit cubic splines to psin

    R = np.linspace(np.min(Rold), np.max(Rold), int(resol))
    Y = np.linspace(np.min(Yold), np.max(Yold), int(resol))

    Z = interp2D(R, Y, Rold, Yold, Zold)

    # ~~~~~~~ Find contour of psiN_boundary

    [Rg, Yg] = np.meshgrid(R, Y)

    fig, ax = plt.subplots()
    cs = ax.contour(Rg, Yg, Z, resol, levels=[psiN_boundary])

    Rpsi, Ypsi = [], []
    for k in range(len(cs.allsegs[0])):
        cc = cs.allsegs[0][k]
        Rb, Yb = [], []
        for i in cc:
            Rb.append(i[0])
            Yb.append(i[1])
        Rpsi.append(np.array(Rb))
        Ypsi.append(np.array(Yb))

    plt.close(fig)

    return Rpsi, Ypsi


def characteristicTime(
    t, x, z, tsaws, rhos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):
    return characteristicTimeSawtoothPhase(t, x, z, tsaws, rhos=rhos)


def characteristicTimeSawtoothPhase(
    t, x, z, tsaws, rhos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):
    """
    This routine calculates
    """

    # ---------------------------------------------------
    # Downsample space to user-specified rhos array
    # ---------------------------------------------------

    zn = []
    for ir in rhos:
        zn.append(z[:, np.argmin(np.abs(ir - x))])
    z = np.transpose(np.array(zn))

    t_orig = copy.deepcopy(t)
    z_orig = copy.deepcopy(z)

    # ---------------------------------------------------
    # Smooth to the same phase in sawtooth period
    # ---------------------------------------------------

    fractionSawtooth = (
        0.2  # Fraction of the sawtooth period to calculate mean before crash (20%)
    )
    Delta_tBefore = np.mean(np.diff(np.unique(tsaws))) * fractionSawtooth

    # Smooth through sawtooth
    zn = []
    for ix in range(len(rhos)):
        z0, t = smoothThroughSawtooth(
            t_orig, z[:, ix], tsaws, 2, Delta_tBefore=Delta_tBefore
        )
        zn.append(z0)
    z = np.transpose(np.array(zn))

    # Calculate maximum difference between sawtooth (maximum of all radii)
    t = t[1:]
    zdiff_max = np.zeros(len(t))
    for ix in range(len(rhos)):
        dd = np.abs(np.diff(z[:, ix]) / z[:-1, ix])
        for it in range(len(t)):
            zdiff_max[it] = np.max([dd[it], zdiff_max[it]])

    # Percent
    zdiff_max = np.array(zdiff_max) * 100.0

    return t, zdiff_max


def findBoundaryMath(
    Rold,
    Yold,
    Zold,
    psiN_boundary,
    resol,
    downsampleNum,
    enforcePosition,
    plotDebug,
    AsymmetryTolerance,
):
    """
    psiN_boundary=0.99999,resol=5E3,downsampleNum = 500,
                                enforcePosition=None,plotDebug = False, AsymmetryTolerance = 0.1
    """

    BigAsymmetry = 0.5  # Regardless of AsymmetryTolerance, if the boundary is 0.5m asymmetric, something cleary
    # has gone wrong. Likely because it got the divertor legs and de geometry does not close

    print(f" \t\t--> Finding g-file flux-surface with psiN = {psiN_boundary}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~ Find contour of psiN_boundary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Rpsi, Ypsi = drawContours(
        Rold, Yold, Zold, resol=resol, psiN_boundary=psiN_boundary
    )

    print(" \t\t\t--> Contours created")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~ From the set of possible contours, find the correct one
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    minn, R_min = 1e5, 0.35
    ind = None
    for k in range(len(Rpsi)):
        if (
            np.abs(np.array(Ypsi[k])).min() < minn
            and np.abs(np.array(Rpsi[k])).min() > R_min
        ):
            minn = np.abs(np.array(Ypsi[k])).min()
            ind = k

    # From those contours, the core one is the one closer to the horizontal axis
    RpsiAll = Rpsi
    YpsiAll = Ypsi
    if enforcePosition is None:
        Rpsi = RpsiAll[ind]
        Ypsi = YpsiAll[ind]
    else:
        Rpsi = RpsiAll[enforcePosition]
        Ypsi = YpsiAll[enforcePosition]

    if plotDebug:
        fig, ax = plt.subplots()
        for i in range(len(RpsiAll)):
            ax.scatter(RpsiAll[i], YpsiAll[i], label=str(i))
        ax.legend()
        print("Chosen contour:" + str(ind))
        plt.show()
        embed()

    print(" \t\t\t--> Contours selected")

    # Sometimes, this method also grabs parts that should not be
    maxx = np.abs(np.max(Ypsi))
    minn = np.abs(np.min(Ypsi))

    if np.abs(maxx - minn) < BigAsymmetry:
        ToleranceOffset = 0.002  # in m
        if np.abs(maxx - minn) > AsymmetryTolerance:
            print(
                " >>>> There is a problem with the contour-finding routine, attempting to correct now..."
            )
            maxT = np.min([maxx, minn]) + ToleranceOffset
            Rpsi = Rpsi[Ypsi > -maxT]
            Ypsi = Ypsi[Ypsi > -maxT]
            Rpsi = Rpsi[Ypsi < maxT]
            Ypsi = Ypsi[Ypsi < maxT]

        # Delete to release memory
        del RpsiAll
        del YpsiAll

        # Round to three decimal places because this is what the next routines to write BOUNDARY files will do
        for i in range(len(Rpsi)):
            Rpsi[i] = round(Rpsi[i], 3)
            Ypsi[i] = round(Ypsi[i], 3)

        # Remove repetitions
        Rbn = []
        Ybn = []
        for i in range(len(Rpsi)):
            if i < len(Rpsi) - 1:
                if Rpsi[i + 1] != Rpsi[i] and Ypsi[i + 1] != Ypsi[i]:
                    Rbn.append(Rpsi[i])
                    Ybn.append(Ypsi[i])
            else:
                if Rpsi[0] != Rpsi[i] and Ypsi[0] != Ypsi[i]:
                    Rbn.append(Rpsi[i])
                    Ybn.append(Ypsi[i])

        # Downsample
        if downsampleNum is not None:
            Rb, Yb = downsampleCurve(Rpsi, Ypsi, nsamp=downsampleNum)
        else:
            Rb, Yb = Rpsi, Ypsi

    else:
        newPsi = 0.9999
        print(
            " >>>> Asymmetry way too big, probably it is not a closed contour, decreasing psiN value to {0}...".format(
                newPsi
            )
        )
        Rb, Yb = findBoundaryMath(
            Rold,
            Yold,
            Zold,
            newPsi,
            resol,
            downsampleNum,
            enforcePosition,
            plotDebug,
            AsymmetryTolerance,
        )

    return Rb, Yb
