import torch, datetime
import matplotlib.pyplot as plt
from mitim_modules.powertorch.physics import CALCtools
from mitim_tools.misc_tools import GRAPHICStools, IOtools, PLASMAtools

from mitim_tools.popcon_tools.FunctionalForms import (
    parabolic,
    PRFfunctionals_Hmode,
    PRFfunctionals_Lmode,
)
from mitim_tools.popcon_tools.aux import FUNCTIONALScalc

T_avol = 5.0
n_avol = 1.2
nu_T = 2.5
nu_n = 1.55

plt.ion()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

x, T = parabolic(Tbar=T_avol, nu=nu_T)
x, n = parabolic(Tbar=n_avol, nu=nu_n)

axs[0, 0].plot(x, T, "-", c="b", label="Parabolic")
aLT = CALCtools.produceGradient(torch.from_numpy(x), torch.from_numpy(T)).numpy()
axs[1, 0].plot(x, aLT, "-", c="b")

print(f"<T> Parabolic = {FUNCTIONALScalc.calculate_simplified_volavg(x,T)[0]:.3f}keV")

# axs[0,1].plot(x,n,'-',c='b')
# aLn = CALCtools.produceGradient(torch.from_numpy(x),torch.from_numpy(n)).numpy()
# axs[1,1].plot(x,aLn,'-',c='b')

# print(f'<n> Parabolic = {FUNCTIONALScalc.calculate_simplified_volavg(x,n)[0]:.3f}')

# x, T, n = PRFfunctionals_Hmode( T_avol, n_avol, nu_T, nu_n, aLT = 2.0 )

# axs[0,0].plot(x,T,'-',c='r',label='PRFfunctionals (H-mode)')
# aLT = CALCtools.produceGradient(torch.from_numpy(x),torch.from_numpy(T)).numpy()
# axs[1,0].plot(x,aLT,'-',c='r')

# print(f'<T> PRF H = {FUNCTIONALScalc.calculate_simplified_volavg(x,T)[0]:.3f}keV')

# axs[0,1].plot(x,n,'-',c='r')
# aLn = CALCtools.produceGradient(torch.from_numpy(x),torch.from_numpy(n)).numpy()
# axs[1,1].plot(x,aLn,'-',c='r')

# print(f'<n> PRF H = {FUNCTIONALScalc.calculate_simplified_volavg(x,n)[0]:.3f}')

timeBeginning = datetime.datetime.now()
x, T, n = PRFfunctionals_Lmode(T_avol, n_avol, nu_n)
print("\t* Took: " + IOtools.getTimeDifference(timeBeginning))

axs[0, 0].plot(x, T, "-", c="g", label="Piecewise linear gradient")
aLT = CALCtools.produceGradient(torch.from_numpy(x), torch.from_numpy(T)).numpy()
axs[1, 0].plot(x, aLT, "-", c="g")

print(f"<T> PRF L = {FUNCTIONALScalc.calculate_simplified_volavg(x,T)[0]:.3f}keV")

axs[0, 1].plot(x, n, "-", c="g")
aLn = CALCtools.produceGradient(torch.from_numpy(x), torch.from_numpy(n)).numpy()
axs[1, 1].plot(x, aLn, "-", c="g")

print(f"<n> PRF L = {FUNCTIONALScalc.calculate_simplified_volavg(x,n)[0]:.3f}")

ax = axs[0, 0]
ax.set_xlabel("$r/a$")
ax.set_ylabel("$T$ (keV)")
ax.set_xlim([0, 1])
ax.set_ylim(bottom=0)
GRAPHICStools.addDenseAxis(ax)
ax.legend()
ax = axs[1, 0]
ax.set_xlabel("$r/a$")
ax.set_ylabel("$a/L_T$")
ax.set_xlim([0, 1])
ax.set_ylim([0, 20])
GRAPHICStools.addDenseAxis(ax)
ax = axs[0, 1]
ax.set_xlabel("$r/a$")
ax.set_ylabel("$n$ ($10^{20}m^{-3}$)")
ax.set_xlim([0, 1])
ax.set_ylim(bottom=0)
GRAPHICStools.addDenseAxis(ax)
ax = axs[1, 1]
ax.set_xlabel("$r/a$")
ax.set_ylabel("$a/L_n$")
ax.set_xlim([0, 1])
ax.set_ylim([0, 20])
GRAPHICStools.addDenseAxis(ax)
