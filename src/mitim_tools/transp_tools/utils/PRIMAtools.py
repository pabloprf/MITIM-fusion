import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

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
