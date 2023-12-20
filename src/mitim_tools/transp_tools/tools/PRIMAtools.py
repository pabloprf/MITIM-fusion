import os
import numpy as np
import matplotlib.pyplot as plt

from mitim_tools.misc_tools import MATHtools, IOtools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print

from IPython import embed


def introducePulseSimulation(
    folder_sim,
    folder_new,
    timeExtract,
    simname="Z17",
    dt=1e-3,
    Strengths=[50.0],
    mu_rhos=[0.1],
    mu_ts=[0.35],
    sigma_rhos=[0.05],
    sigma_ts=[0.01],
    skew_rhos=[0.0],
    skew_ts=[0.0],
):
    """
    folder_sim contains files of previous simulation to start with
    folder_new is the new simulation folder

    """

    s = CONFIGread.load_settings()
    user = s[s["preferences"]["ntcc"]]["username"]

    if np.min(mu_ts) - timeExtract < 0.1:
        print(
            "Pulses are introduced less than 100ms after extraction, sure? (c)",
            typeMsg="q",
        )

    folder_new = IOtools.expandPath(folder_new) + "/"

    IOtools.askNewFolder(folder_new)

    # ------------------------------------------------------
    # Get all input files from previous run
    # ------------------------------------------------------

    runid = IOtools.findExistingFiles(folder_sim, "TR.DAT")[0].split("TR.DAT")[0]
    shot = runid[:5]

    os.system(f"cp {folder_sim}/PRF{shot}.* {folder_new}/")
    os.system(f"cp {folder_sim}/{runid}TR.DAT {folder_new}/")

    namelist_file = f"{folder_new}/{runid}TR.DAT"
    cdf_file = f"{folder_sim}/{runid}.CDF"

    # ------------------------------------------------------
    # Introduce pulses
    # ------------------------------------------------------

    addPulsesToRun(
        namelist_file,
        dt=dt,
        Strengths=Strengths,
        mu_rhos=mu_rhos,
        mu_ts=mu_ts,
        sigma_rhos=sigma_rhos,
        sigma_ts=sigma_ts,
        skew_rhos=skew_rhos,
        skew_ts=skew_ts,
    )

    # ------------------------------------------------------
    # Update files
    # ------------------------------------------------------

    UFILEStools.updateTypicalFiles(folder_new, cdf_file, timeExtract, shot=shot)

    t1, t2 = timeExtract, 100.0

    extra = 0.0
    os.system("mv {0} {0}_old".format(namelist_file))
    with open(namelist_file + "_old", "r") as f:
        aux = f.readlines()
    with open(namelist_file, "w") as f:
        for line in aux:
            if "~update" not in line:
                lineCopy = line
            else:
                lineCopy = f"~update_time={t1 + 0.01 + extra}\n"
                extra += 0.001
            f.write(lineCopy)

    _ = IOtools.changeValue(namelist_file, "tinit", t1, [""], "=")
    _ = IOtools.changeValue(namelist_file, "ftime", t2, [""], "=")

    nameRunTot = f"{shot}{simname}"
    os.system(f"mv {namelist_file} {folder_new}/{nameRunTot}TR.DAT")

    from im_tools.modules import TRANSPmodule

    TRANSPmodule.launchTRANSP_std(
        nameRunTot,
        folder_new,
        "D3D",
        loc,
        "tshare",
        {"trmpi": 32, "toricmpi": 1, "ptrmpi": 32},
        email,
        label="PRF, single run",
    )


def introduceRadiationPulses(
    PulseParameters_list,
    timeRange=[0, 100],
    mindt=1e-3,
    ufile="PRF12345.LHE",
    ufileType="lhe",
    cdffile=None,
    timeExtraction=None,
    plot=False,
):
    maxSigma_times = 5

    min_time, max_time = np.inf, -np.inf
    for PulseParameters in PulseParameters_list:
        if PulseParameters[2] - PulseParameters[4] * maxSigma_times < min_time:
            min_time = PulseParameters[2] - PulseParameters[4] * maxSigma_times
        if PulseParameters[2] + PulseParameters[4] * maxSigma_times > max_time:
            max_time = PulseParameters[2] + PulseParameters[4] * maxSigma_times

    if cdffile is not None:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~ Extract radiation from a CDF file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        from transp_tools import CDFtools

        cdf = CDFtools.CDFreactor(cdffile)
        it = np.argmin(np.abs(cdf.t - timeExtraction))

        rho = cdf.x[it]
        Prad = cdf.Prad[it]  # MW/m^3 or W/cm^3

        ufileN = "bol"

        t = np.linspace(0, 100, int(1e4))

    else:
        t = np.arange(
            min_time, max_time, mindt / 2
        )  # linspace(mu_t-sigma_t*10,mu_t+sigma_t*10,int(1E4))
        rho = np.linspace(0, 1, 1000)
        Prad = np.zeros((len(t), len(rho)))

        ufileN = ufileType

    for PulseParameters in PulseParameters_list:
        # Parameters pulse

        Strength = PulseParameters[0]
        mu_rho = PulseParameters[1]
        mu_t = PulseParameters[2]
        sigma_rho = PulseParameters[3]
        sigma_t = PulseParameters[4]
        skew_rho = PulseParameters[5]
        skew_t = PulseParameters[6]
        speed = 0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~ Create pulse
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        pulse = MATHtools.perturbativePulse(
            rho, t, Strength, mu_rho, speed, mu_t, sigma_rho, sigma_t, skew_rho, skew_t
        )

        Prad += pulse

    if plot:
        plt.ion()
        fig, axs = plt.subplots(nrows=2)
        for PulseParameters in PulseParameters_list:
            mu_t = PulseParameters[2]
            it = np.argmin(np.abs(t - mu_t))
            axs[0].plot(rho, Prad[it])
        axs[0].set_xlabel("$\\rho_N$")
        for PulseParameters in PulseParameters_list:
            mu_rho = PulseParameters[1]
            ix = np.argmin(np.abs(rho - mu_rho))
            axs[1].plot(t, Prad[:, ix])
        axs[1].set_xlabel("Time (s)")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~ Write radiation to ufile
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    UF = UFILEStools.UFILEtransp(scratch=ufileN, rhos=rho)

    UF.Variables["X"] = rho
    UF.Variables["Y"] = t
    UF.Variables["Z"] = Prad

    UF.writeUFILE(ufile)


def addPulsesToRun(
    namelist_file,
    dt=1e-3,
    Strengths=[50.0],
    mu_rhos=[0.1],
    mu_ts=[0.35],
    sigma_rhos=[0.05],
    sigma_ts=[0.01],
    skew_rhos=[0.0],
    skew_ts=[0.0],
):
    """
    Strength is the peak in MW/m^3
    """

    if len(namelist_file.split("/")) == 1:
        namelist_file = "./" + namelist_file

    namelist_file = IOtools.expandPath(namelist_file)

    folder, _ = IOtools.reducePathLevel(namelist_file, level=1, isItFile=True)

    Pulses = []
    for Strength, mu_rho, mu_t, sigma_rho, sigma_t, skew_rho, skew_t in zip(
        Strengths, mu_rhos, mu_ts, sigma_rhos, sigma_ts, skew_rhos, skew_ts
    ):
        Pulses.append([Strength, mu_rho, mu_t, sigma_rho, sigma_t, skew_rho, skew_t])

    introduceRadiationPulses(
        Pulses, ufile=folder + "PRF12345.LHE", ufileType="lhe", plot=True, mindt=dt
    )
    introduceRadiationPulses(
        [[0.0, mu_rho, mu_t, sigma_rho, sigma_t, skew_rho, skew_t]],
        ufile=folder + "PRF12345.LHJ",
        ufileType="lhj",
        mindt=dt,
    )

    TRANSPvars = {
        "NLLH": "T",
        "NANTLH": "1",
        "fghzLH": "1.0",
        "dtlh": dt,
        "dtmaxg": dt,
        "dtbeam": dt,
        "dticrf": dt,
        "TLHON": "0.0",
        "TLHOFF": "100.0",
        "POWRLHAN": "1.0",
        "NLLHUDAT": "T",
        "prelhe": '"PRF"',
        "extlhe": '"LHE"',
        "nrilhe": "-5",
        "prelhj": '"PRF"',
        "extlhj": '"LHJ"',
        "nrilhj": "-5",
    }

    for itag in TRANSPvars:
        _ = IOtools.changeValue(namelist_file, itag, TRANSPvars[itag], "", "=")

    nameBaseShot = namelist_file.split("/")[-1].split("TR")[0][:-3]

    # Change shot number
    IOtools.changeValue(namelist_file, "nshot", nameBaseShot, None, "=")

    # Add inputdir to namelist
    _ = IOtools.changeValue(
        namelist_file, "inputdir", folder, "", "=", CommentChar=None
    )


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


"""
shotChiPertpy.py
Alex Creely
17.4.17

Code to extract all required data from a shot to calculate perturbative
diffusivity.

python shotChiPertpy.py shot pulseTimes inChannel outChannel timeRange smooth
    backgroundAvg

"""


# Object class that holds all information about calculation
class chiPertCalculator(object):
    def __init__(self, cdf, pulseTimes, rhoRange, timeRange, numChan=10):
        self.pulseTimes = pulseTimes

        self.numChan = numChan
        self.rhosC = np.linspace(rhoRange[0], rhoRange[1], self.numChan)
        self.inChannel = 1
        self.outChannel = self.numChan

        self.timeRange = timeRange

        self.backgroundAvg = 1

        self.numPulses = len(pulseTimes)

        eleTempTime = []
        radiusTime = []
        eleTemp = []
        gradTemp = []
        radius = []
        for i in self.rhosC:
            ir = np.argmin(np.abs(cdf.x_lw - i))
            eleTemp.append(cdf.Te[:, ir])
            gradTemp.append(cdf.aLTe[:, ir])
            radius.append(cdf.rmaj_LFx[:, ir])
            eleTempTime.append(cdf.t)
            radiusTime.append(cdf.t)
        self.eleTemp = np.array(eleTemp)
        self.gradTemp = np.array(gradTemp)
        self.radiusDiag = np.array(radius)
        self.eleTempTime = np.array(eleTempTime)
        self.radiusDiagTime = np.array(radiusTime)

        it = np.argmin(np.abs(cdf.t - pulseTimes[0]))
        self.a = cdf.a[it]
        self.kappa = cdf.kappa[it]
        self.s = cdf.ShafShift[it]

    def prepareData(self, axs=None):
        self.plotPulse = axs

        print("    Preparing Data for Diffusivity Calculation")

        # create empty arrays to hold the time, amplitude and radius
        # of measurement channels for calculation of chiPert
        self.tPulse = np.zeros([self.numPulses, self.numChan])
        self.aPulse = np.zeros([self.numPulses, self.numChan])
        self.rPulse = np.zeros([self.numPulses, self.numChan])

        # loop through pulses j
        for j in range(0, self.numPulses):
            indexStart = (np.abs(self.pulseTimes[j] - self.eleTempTime)).argmin()

            indexEnd = (
                np.abs((self.pulseTimes[j] + self.timeRange) - self.eleTempTime)
            ).argmin()

            # temporary variables for storage
            tPeak = np.zeros(self.numChan)
            amp = np.zeros(self.numChan)
            rad = np.zeros(self.numChan)

            # loop through channels
            rr = range(self.numChan)

            for i in rr:
                # Generate smoothed temperature data (get rid of bit noise, but not needed for TRANSP)
                smoothedTe = self.eleTemp[i, indexStart:indexEnd]
                smoothedgradTe = self.gradTemp[i, indexStart:indexEnd]

                # The peak temperature
                peakTe = np.amax(smoothedTe)

                # The background temperature
                backgroundTe = np.mean(smoothedTe[0 : self.backgroundAvg])
                backgroundgradTe = np.mean(smoothedgradTe[0 : self.backgroundAvg])

                # index of peak temperature
                peakIndex = np.argmax(smoothedTe)

                # time for peak temperature
                peakTime = self.eleTempTime[i, peakIndex + indexStart]

                # Plot pulses if desired
                if self.plotPulse is not None:
                    ax = self.plotPulse

                    ax[0].plot(
                        self.eleTempTime[i, indexStart:indexEnd] * 1000,
                        (self.eleTemp[i, indexStart:indexEnd] - backgroundTe) * 1000,
                        color="blue",
                        linewidth=1,
                    )

                    ax[1].plot(
                        self.eleTempTime[i, indexStart:indexEnd] * 1000,
                        (self.gradTemp[i, indexStart:indexEnd] - backgroundgradTe)
                        / backgroundgradTe
                        * 100.0,  # should this be 1000
                        color="blue",
                        linewidth=1,
                    )

                    ax[0].set_title(f"$\\Delta T_e$, t={str(self.pulseTimes[j])}s")
                    ax[0].set_ylabel("Pulse Amplitude (eV)")
                    ax[0].set_xlabel("Time (ms)")
                    # ax[0].tick_params(labelsize=18)
                    # ax[0].legend(fontsize=16,loc='best')

                    ax[1].set_title(f"$\\Delta a/L_T$, t={str(self.pulseTimes[j])}s")
                    ax[1].set_ylabel("Pulse Amplitude (%)")
                    ax[1].set_xlabel("Time (ms)")
                    # ax[1].tick_params(labelsize=18)
                    # ax[1].legend(fontsize=16,loc='best')

                # Find the index for the radius
                radiusIndex = (np.abs(self.radiusDiagTime[i, :] - peakTime)).argmin()

                # look up the radius of a given channel
                # Use radius mapped to magnetic axis

                # radius = self.radiusDiagRmagAxis[i,radiusIndex]    ########### ########### ########### ########### ########### ########### ###########
                radius = self.radiusDiag[i, radiusIndex]

                # store calculated peaks in temporary variables
                tPeak[i - self.inChannel + 1] = peakTime
                amp[i - self.inChannel + 1] = peakTe - backgroundTe
                rad[i - self.inChannel + 1] = radius

            # Store peaks in final variables, for all pulses
            self.tPulse[j, :] = tPeak
            self.aPulse[j, :] = amp
            self.rPulse[j, :] = rad

        print("    Data Prepared")

        return

    # run through all pulses in shot, and calculate uncertainty
    def calculateShotChiPert(self):
        print("    Calculating Perturbative Diffusivity")

        # Calculate perturbative diffusivity for the whole shot

        self.chiPert = self.calcChiPert(self.tPulse, self.aPulse, self.rPulse)

        # Calculate perturbative diffusivity for individual pulses

        self.chiPertPulse = np.zeros(self.numPulses)

        for i in range(0, self.numPulses):
            self.chiPertPulse[i] = self.calcChiPert(
                self.tPulse[i, :], self.aPulse[i, :], self.rPulse[i, :]
            )

        self.expErrorPct = 0.0
        self.stdError = np.std(self.chiPertPulse) / np.sqrt(self.numPulses)
        self.expError = self.expErrorPct * self.chiPert
        self.totError = np.sqrt(self.stdError**2 + (self.expError) ** 2)

        print("    Perturbative Diffusivity Calculated")

    def calculateCompositeSawtooth(self):
        print("    Calculating Composite Sawtooth")

        # indices just to determine index length of one pulse window
        index1 = (np.abs(self.pulseTimes[0] - self.eleTempTime)).argmin()

        index2 = (
            np.abs((self.pulseTimes[0] + self.timeRange) - self.eleTempTime)
        ).argmin()

        indexLength = index2 - index1

        # fraction of window before and after peak
        lengthBefore = int(0.4 * indexLength)

        lengthAfter = indexLength - lengthBefore

        # number of channels
        numChannels = self.outChannel - self.inChannel + 1

        # Define empty variables
        self.compositeTemp = np.zeros([indexLength, numChannels])
        self.compositeTempSmooth = np.zeros([indexLength, numChannels])
        self.compositeTime = np.zeros([indexLength])
        self.compositeRadius = np.zeros([numChannels])

        # Fill time with time variable
        self.compositeTime = (
            self.eleTempTime[0, index1:index2] - self.eleTempTime[0, index1]
        )

        # Iterate through pulses
        for j in range(0, self.numPulses):
            # Synchronize with peak of innermost channel ***FIX***

            indexBeginSearch = (np.abs(self.pulseTimes[j] - self.eleTempTime)).argmin()

            indexEndSearch = (
                np.abs((self.pulseTimes[j] + self.timeRange) - self.eleTempTime)
            ).argmin()

            indexRawPeak = (
                np.argmax(self.eleTemp[0, indexBeginSearch:indexEndSearch])
                + indexBeginSearch
            )

            """peakIndex = indexRawPeak

            """
            # peakIndex = np.abs()
            peakIndex = np.abs(self.tPulse[j, 0] - self.eleTempTime[0, :]).argmin()

            indexStart = peakIndex - lengthBefore

            indexEnd = peakIndex + lengthAfter

            # loop through channels
            for i in range(self.inChannel - 1, self.outChannel):
                # Channel index
                chanIndex = i - self.inChannel + 1

                # Calculate average Te to subtract off
                backgroundTe = np.mean(
                    self.eleTemp[i, indexStart : (indexStart + self.backgroundAvg)]
                )

                # Add temperatures from each pulse to the composite temp
                self.compositeTemp[:, chanIndex] = self.compositeTemp[:, chanIndex] + (
                    self.eleTemp[i, indexStart:indexEnd] - backgroundTe
                )
                # Calcualte the index for the radius
                radiusIndex = (
                    np.abs(self.radiusDiagTime[i, :] - self.tPulse[j, chanIndex])
                ).argmin()

                # Add radius from each pulse to composite radius
                self.compositeRadius[chanIndex] = (
                    self.compositeRadius[chanIndex] + self.radiusDiag[i, radiusIndex]
                )  ########### ########### ########### ########### ########### ########### ###########

        # Divide composites by the number of pulses
        self.compositeTemp = self.compositeTemp / self.numPulses
        self.compositeRadius = self.compositeRadius / self.numPulses

        # Begin chiPert calc for composite pulse

        self.tPulseComposite = np.zeros([self.numChan])
        self.aPulseComposite = np.zeros([self.numChan])
        self.rPulseComposite = np.zeros([self.numChan])

        # Smooth composite pulse
        from statsmodels.nonparametric.smoothers_lowess import lowess

        for i in range(0, numChannels):
            self.compositeTempSmooth[:, i] = lowess(
                self.compositeTemp[:, i],
                self.compositeTime,
                is_sorted=True,
                frac=(0.1 + (i * 0.0 + self.eps005)),
                it=0,
                return_sorted=False,
            )

            # Peak temperature
            peakTeSmooth = np.amax(self.compositeTempSmooth[:, i])

            # index and time for peak temperature
            peakIndexSmooth = np.argmax(self.compositeTempSmooth[:, i])
            peakTime = self.compositeTime[peakIndexSmooth]

            self.tPulseComposite[i] = peakTime
            self.aPulseComposite[i] = peakTeSmooth
            self.rPulseComposite[i] = self.compositeRadius[i]

        # calculate composite chi pert
        self.chiPertComposite = self.calcChiPert(
            self.tPulseComposite, self.aPulseComposite, self.rPulseComposite
        )

        if self.plotPulse:
            # Plot to check
            plt.figure(figsize=(8, 6))

            for i in range(0, numChannels):
                if i == 0:
                    plt.plot(
                        self.compositeTime * 1000,
                        self.compositeTemp[:, i],
                        linewidth=3,
                        color="blue",
                        label="Composite Data",
                    )

                else:
                    plt.plot(
                        self.compositeTime * 1000,
                        self.compositeTemp[:, i],
                        linewidth=3,
                        color="blue",
                    )

            for i in range(0, numChannels):
                if i == 0:
                    plt.plot(
                        self.compositeTime * 1000,
                        self.compositeTempSmooth[:, i],
                        linewidth=3,
                        color="red",
                        linestyle="--",
                        label="Smoothed Data",
                    )

                else:
                    plt.plot(
                        self.compositeTime * 1000,
                        self.compositeTempSmooth[:, i],
                        linewidth=3,
                        color="red",
                        linestyle="--",
                    )

            axes = plt.gca()
            # axes.set_title('Composite Sawtooth')
            # axes.set_xlim([0,180])
            # axes.set_ylim([-0.01,.1])
            axes.set_ylabel("Pulse Amplitude (eV)", fontsize=20)
            axes.set_xlabel("Time (ms)", fontsize=20)
            axes.tick_params(labelsize=18)

            plt.legend(fontsize=16, loc="best")

        print("    Composite Sawtooth Calculated")

        return

    def calcChiPert(self, tPulse, aPulse, rPulse):
        # The number of radii in the measuremnet
        if np.array(tPulse.shape).shape[0] > 1:
            # The number of heat pulses to average over
            Npulse = tPulse.shape[0]
            Nradii = tPulse.shape[1]

        else:
            Npulse = 1
            Nradii = tPulse.shape[0]

        # arrays of the slopes of the radius against time and against amplitude
        tSlope = np.zeros([Npulse])
        aSlope = np.zeros([Npulse])

        # iterate through pulses, finding the slope of radius agains time

        if Npulse == 1:
            try:
                fit = np.polyfit(rPulse[:], tPulse[:], 1)
            except:
                fit = np.polyfit(rPulse[0], tPulse[0], 1)
            tSlope[0] = fit[0]

        else:
            for i in range(0, Npulse):
                fit = np.polyfit(rPulse[i, :], tPulse[i, :], 1)
                tSlope[i] = fit[0]

        # Average over pulses
        tSlopeAverage = (1 / float(Npulse)) * (np.sum(tSlope))

        # Uncorrected pulse velocity
        vPdag = 1 / tSlopeAverage
        print(vPdag)

        # Correct for curvature and Shafranov shift
        vP = np.sqrt(self.kappa) * ((self.a) / (self.a - self.s)) * vPdag

        # Array of logarithms of amplitudes (*1000 keV to eV)
        logApulse = np.log10(aPulse * 1000)

        # Calculate for each pulse
        if Npulse == 1:
            try:
                fit = np.polyfit(rPulse[:], logApulse[:], 1)
            except:
                fit = np.polyfit(rPulse[0], logApulse[0], 1)
            aSlope[0] = fit[0]

        else:
            for i in range(0, Npulse):
                fit = np.polyfit(rPulse[i, :], logApulse[i, :], 1)
                aSlope[i] = fit[0]

        # Average over pulses
        aSlopeAverage = (1 / float(Npulse)) * (np.sum(aSlope))

        # Calculated uncorrected alpha radial damping parameter (in dB)
        alphaDag = np.abs(10 * self.a * aSlopeAverage)

        # Correct for Shafranov Shift
        alpha = ((self.a - self.s) / self.a) * alphaDag

        # Calculate the perturbative thermal diffusivity
        chiPert = 4.2 * ((self.a * np.sqrt(self.kappa)) * vP) / alpha

        return chiPert

    def printOutputs(self):
        print(
            "    Chi^pert =               "
            + str(np.around(self.chiPert, decimals=2))
            + " m^2/s +- "
            + str(np.around(self.totError, decimals=2))
            + " m^2/s"
        )

        return self.chiPert
