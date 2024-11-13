import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


# Object class that holds all information about calculation
class chiPertCalculator(object):
    def __init__(self, cdf, pulseTimes, rhoRange, timeRange, plotBool):
        self.cdf = cdf

        self.pulseTimes = pulseTimes
        self.timeRange = timeRange
        self.plotPulse = plotBool

        ## This is supposed to be the toroidal flux coordinate at the last sawtooth? But I don't have sawteeth?
        it = np.argmin(np.abs(cdf.t - pulseTimes[0]))
        x_lw = cdf.rho_tor[it, :]

        # Adaptation for using the number of channels available in the simulation
        self.numPulses = 1  # I am hard coding this in
        ix1 = np.argmin(np.abs(x_lw - rhoRange[0]))
        ix2 = np.argmin(np.abs(x_lw - rhoRange[1]))
        self.rhosC = x_lw[
            ix1 : ix2 + 1
        ]  # np.linspace(rhoRange[0],rhoRange[1],self.numChan)
        self.numChan = len(self.rhosC)
        print("Number of Channels: " + str(self.numChan))

        self.inChannel = 1
        self.outChannel = self.numChan

        self.eleTemp = None
        self.eleTempTime = None
        self.radiusDiag = None
        self.radiusDiagTime = None
        self.zDiag = None

        self.a = None
        self.elong = None
        self.s = None
        self.q = None
        self.qpsi = None
        self.avgq = None

        self.dcnDen = None
        self.avgTemp = None
        self.avgGradT = None

        self.RmajIn = None
        self.RmajOut = None
        self.rhoTorIn = None
        self.rhoTorOut = None

        self.radiusDiagRmagAxis = None
        self.teRhoPol = None
        self.rMinorAvg = None
        self.rMagAxAvg = None

        self.expErrorPct = None

        self.tPulse = None
        self.aPulse = None
        self.rPulse = None

        self.chiPert = None
        self.chiPertPulse = None
        self.chiPertComposite = None
        self.stdError = None
        self.expError = None
        self.totError = None

        # get radius from CDF--> assumes Pablo's code is right
        # self.rhosC                  = np.linspace(rhoRange[0],rhoRange[1],self.numChan)
        eleTemp = []
        eleTempTime = []
        radiusTime = []
        eleTemp = []
        gradTemp = []
        radius = []
        for i in self.rhosC:
            ir = np.argmin(np.abs(x_lw - i))
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

        # get temperature from CDF --> assumes Pablo's code is right
        self.a = cdf.a[it]
        self.elong = cdf.elong[it]
        self.s = cdf.shift[it]

        return

    def prepareData(self):
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
            for i in range(self.numChan):
                # Generate smoothed temperature data (get rid of bit noise)

                # Smoothing not needed in TRANSP
                smoothedTe = self.eleTemp[i, indexStart:indexEnd]
                smoothedgradTe = self.gradTemp[i, indexStart:indexEnd]

                # The peak temperature
                peakTe = np.amax(smoothedTe)

                # The background temperature
                self.backgroundAvg = 1  # value Pablo set
                self.backgroundTe = np.mean(smoothedTe[0 : self.backgroundAvg])

                # index of peak temperature
                peakIndex = np.argmax(smoothedTe)

                # time for peak temperature
                peakTime = self.eleTempTime[i, peakIndex + indexStart]

                # this is where the plotting routine was before

                # Find the index for the radius
                radiusIndex = (np.abs(self.radiusDiagTime[i, :] - peakTime)).argmin()

                # look up the radius of a given channel
                # Use radius mapped to magnetic axis
                radius = self.radiusDiag[i, radiusIndex]

                # store calculated peaks in temporary variables
                tPeak[i - self.inChannel + 1] = peakTime
                amp[i - self.inChannel + 1] = peakTe - self.backgroundTe
                rad[i - self.inChannel + 1] = radius

            # Store peaks in final variables, for all pulses
            self.tPulse[j, :] = tPeak
            self.aPulse[j, :] = amp
            self.rPulse[j, :] = rad

        print("    Data Prepared")

        return

    def calcChiPert(
        self, tPulse, aPulse, rPulse
    ):  # I'm not sure why this doesn't just look at self
        # The number of radii in the measuremnet
        # if np.array(tPulse.shape).shape[0] > 1:

        # The number of heat pulses to average over
        Npulse = tPulse.shape[0]
        Nradii = tPulse.shape[1]

        # I do not have this formatting for single pulses?
        # else:

        #     Npulse = 1
        #     Nradii = tPulse.shape[0]

        # arrays of the slopes of the radius against time and against amplitude
        tSlope = np.zeros([Npulse])
        aSlope = np.zeros([Npulse])

        # iterate through pulses, finding the slope of radius agains time
        # if Npulse == 1:

        #     fit = np.polyfit(rPulse[:],tPulse[:],1)
        #     tSlope[0] = fit[0]

        # else:

        for j in range(0, Npulse):
            fit = np.polyfit(rPulse[j, :], tPulse[j, :], 1)
            tSlope[j] = fit[0]

            if self.plotPulse:
                plt.ion()
                fig, ax = plt.subplots()
                plt.scatter(
                    self.rPulse[j, :], self.tPulse[j, :]
                )  # map radius to rho tor here!
                plt.plot(rPulse[j, :], fit[1] + fit[0] * rPulse[j, :])

                plt.xlabel("Radius")
                plt.ylabel("Time of max Te")

            # for i in range(self.numChan):

            #     rho = self.rhosC[i]
            #     plt.scatter()

        # Average over pulses
        tSlopeAverage = (1 / float(Npulse)) * (np.sum(tSlope))

        # Uncorrected pulse velocity
        vPdag = 1 / tSlopeAverage
        print("vPdag: " + str(vPdag))

        # Correct for curvature and Shafranov shift
        vP = np.sqrt(self.elong) * ((self.a) / (self.a - self.s)) * vPdag
        print("vP: " + str(vP))

        # Array of logarithms of amplitudes (*1000 keV to eV)
        logApulse = np.log10(aPulse * 1000)

        # Calculate for each pulse
        # if Npulse == 1:

        #     fit = np.polyfit(rPulse[:],logApulse[:],1)
        #     aSlope[0] = fit[0]

        # else:

        # loop through pulses j (use same indexing convention as prepareData to facilitate plotting)
        for j in range(0, Npulse):
            fit = np.polyfit(rPulse[j, :], logApulse[j, :], 1)
            aSlope[j] = fit[0]

            indexStart = (np.abs(self.pulseTimes[j] - self.eleTempTime)).argmin()

            indexEnd = (
                np.abs((self.pulseTimes[j] + self.timeRange) - self.eleTempTime)
            ).argmin()

            if self.plotPulse:
                plt.ion()
                fig, ax = plt.subplots()

            for i in range(self.numChan):
                # Smoothing not needed in TRANSP
                smoothedTe = self.eleTemp[i, indexStart:indexEnd]
                smoothedgradTe = self.gradTemp[i, indexStart:indexEnd]

                # The peak temperature
                peakTe = np.amax(smoothedTe)

                # index of peak temperature
                peakIndex = np.argmax(smoothedTe)

                # time for peak temperature
                peakTime = self.eleTempTime[i, peakIndex + indexStart]

                # Plot pulses if desired
                if self.plotPulse:
                    colors = ["b", "g", "r", "c", "m", "y", "k"]

                    # if i == self.inChannel-1:
                    # plt.plot(self.eleTempTime[i,indexStart:indexEnd]*1000,
                    #     (self.eleTemp[i,indexStart:indexEnd]-backgroundTe),
                    #     color = 'blue',linewidth=2,label = 'Raw Data')

                    # plt.plot(self.eleTempTime[i,indexStart:indexEnd]*1000,
                    #     (smoothedTe - backgroundTe),linewidth=2,
                    #          color='red',linestyle='--',
                    #          label = 'Smoothed Data')

                    if True:
                        plt.plot(
                            self.eleTempTime[i, indexStart:indexEnd] * 1000,
                            (self.eleTemp[i, indexStart:indexEnd] - self.backgroundTe),
                            color=colors[i % 7],
                            linewidth=2,
                            label=str(i),
                        )

                        # plt.plot(self.eleTempTime[i,indexStart:indexEnd]*1000,
                        #     (smoothedTe - backgroundTe),linewidth=2,
                        #          color='red',linestyle='--')

                    axes = plt.gca()
                    axes.set_title(str(self.pulseTimes[j]))
                    axes.set_ylabel("Pulse Amplitude (keV)", fontsize=20)
                    axes.set_xlabel("Time (ms)", fontsize=20)

            if self.plotPulse:
                plt.legend(fontsize=16, loc="best")
                # plt.ylim([0.035, 0.065])
                # plt.xlim([5020, 5028])

        # Average over pulses
        aSlopeAverage = (1 / float(Npulse)) * (np.sum(aSlope))

        # Calculated uncorrected alpha radial damping parameter (in dB)
        alphaDag = np.abs(10 * self.a * aSlopeAverage)

        # Correct for Shafranov Shift
        alpha = ((self.a - self.s) / self.a) * alphaDag
        print("alpha: " + str(alpha))

        # Calculate the perturbative thermal diffusivity
        chiPert = 4.2 * ((self.a * np.sqrt(self.elong)) * vP) / alpha
        print("chiPert: " + str(chiPert))

        return chiPert
