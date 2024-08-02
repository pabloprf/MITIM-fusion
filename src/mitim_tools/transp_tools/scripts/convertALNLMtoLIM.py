import numpy as np
import matplotlib.pyplot as plt

from IPython import embed


def findIntersection(x0, y0, t0, x1, y1, t1):
    x_int = ((y0 - y1) + (x0 * np.tan(t0) - x1 * np.tan(t1))) / (
        np.tan(t0) - np.tan(t1)
    )

    y_int = y0 - np.tan(t0) * (x_int - x0)

    return x_int, y_int


def allowPointsBetween(x, y, xp0, xp1):
    dist = np.linalg.norm(xp0 - xp1)

    xn = []
    yn = []
    for i in range(len(x)):
        xp = np.array([x[i], y[i]])
        dist0 = np.linalg.norm(xp - xp0)
        dist1 = np.linalg.norm(xp - xp1)
        if dist0 + dist1 < dist:
            xn.append(x[i])
            yn.append(y[i])

    return np.array(xn), np.array(yn)


x = np.array(
    [
        104.0,
        114.0,
        135.0,
        143.0,
        163.0,
        196.0,
        215.0,
        221.0,
        213.0,
        187.0,
        145.0,
        123.0,
        106.0,
    ]
)
y = np.array(
    [
        0.0,
        70.0,
        110.0,
        114.0,
        114.0,
        78.0,
        39.0,
        0.0,
        -28.0,
        -68.0,
        -106.0,
        -82.0,
        -30.0,
    ]
)
t = (
    np.array(
        [
            90.0,
            105.0,
            130.0,
            146.0,
            20.0,
            60.0,
            70.0,
            90.0,
            105.0,
            130.0,
            0.0,
            61.0,
            82.0,
        ]
    )
    * np.pi
    / 180.0
)


xT = np.array([])
yT = np.array([])
for i in range(len(x)):
    pastP = i - 1
    presentP = i
    nextP = i + 1

    if pastP < 0:
        pastP = len(x) - 1
    if nextP > len(x) - 1:
        nextP = 0

    # Intersection with next and past curve
    x0_int, y0_int = findIntersection(
        x[presentP], y[presentP], t[presentP], x[nextP], y[nextP], t[nextP]
    )
    x1_int, y1_int = findIntersection(
        x[presentP], y[presentP], t[presentP], x[pastP], y[pastP], t[pastP]
    )

    xT = np.append(xT, x1_int)
    yT = np.append(yT, y1_int)

    vec = np.arange(-1000, 1000, 1)
    xn = []
    yn = []
    for j in vec:
        xn.append(x[presentP] + j * np.cos(t[presentP]))
        yn.append(y[presentP] - j * np.sin(t[presentP]))
    xn0 = []
    yn0 = []
    for j in vec:
        xn0.append(x[pastP] + j * np.cos(t[pastP]))
        yn0.append(y[pastP] - j * np.sin(t[pastP]))
    xn1 = []
    yn1 = []
    for j in vec:
        xn1.append(x[nextP] + j * np.cos(t[nextP]))
        yn1.append(y[nextP] - j * np.sin(t[nextP]))
    # xnt,ynt = allowPointsBetween(xn,yn,np.array([x0_int,y0_int]),np.array([x1_int,y1_int]))
    # xT = np.append(xT,xnt); yT = np.append(yT,ynt);

    fig, ax = plt.subplots()
    ax.scatter(x, y, 20)
    ax.scatter([x0_int, x1_int], [y0_int, y1_int], 50)
    ax.plot(xn, yn)
    ax.plot(xn0, yn0)
    ax.plot(xn1, yn1)
    embed()


xT = np.append(xT, xT[0])
yT = np.append(yT, yT[0])

plt.ion()
fig, ax = plt.subplots()
ax.scatter(x, y, 50)
ax.plot(xT, yT)
ax.set_aspect("equal")
