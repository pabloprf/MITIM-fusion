#import required system and numerical routines
import sys
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from IPython import embed
from scipy.optimize import curve_fit

def linear(x, c1, c2): 
    return c1 * x + c2

#Object class that holds all information about calculation
class chiPertCalculator(object):

    def __init__(self, cdf, pulseTime, rhoRange, timeRange, plotBool):

        self.cdf = cdf

        self.pulseTime              = pulseTime
        self.timeRange              = timeRange
        self.plotPulse              = plotBool


        ## This is supposed to be the toroidal flux coordinate at the last sawtooth? But I don't have sawteeth?
        it                          = np.argmin(np.abs(cdf.t-pulseTime))
        x_lw                        = cdf.rho_tor[it, :]

        # Adaptation for using the number of channels available in the simulation
        ix1                         = np.argmin(np.abs(x_lw-rhoRange[0]))
        ix2                         = np.argmin(np.abs(x_lw-rhoRange[1]))
        self.rhosC                  = x_lw[ix1:ix2+1] #np.linspace(rhoRange[0],rhoRange[1],self.numChan)
        self.numChan                = len(self.rhosC)
        print('Number of Channels: ' + str(self.numChan))

        self.eleTemp                = None
        self.eleTempTime            = None
        self.radiusDiag             = None
        self.radiusDiagTime         = None

        self.a                      = None
        self.elong                  = None
        self.s                      = None
        
        self.tPulse                 = None
        self.aPulse                 = None
        self.rPulse                 = None

        eleTemp                     = []
        eleTempTime                 = []
        radiusTime                  = []
        eleTemp                     = []
        gradTemp                    = []
        radius                      = []
        for i in self.rhosC:
            ir = np.argmin(np.abs(x_lw-i))
            eleTemp.append(cdf.Te[:,ir])
            gradTemp.append(cdf.aLTe[:,ir])
            radius.append(cdf.rmaj_LFx[:,ir])
            eleTempTime.append(cdf.t)
            radiusTime.append(cdf.t)
        self.eleTemp                = np.array(eleTemp)
        self.gradTemp               = np.array(gradTemp)
        self.radiusDiag             = np.array(radius)
        self.eleTempTime            = np.array(eleTempTime)
        self.radiusDiagTime         = np.array(radiusTime)

        #get temperature from CDF --> assumes Pablo's code is right
        self.a                      = cdf.a[it]
        self.elong                  = cdf.elong[it]
        self.s                      = cdf.shift[it]
        self.ac                     = self.a*np.sqrt(self.elong)

        return

    def prepareData(self):

        print('    Preparing Data for Diffusivity Calculation')

        #create empty arrays to hold the time, amplitude and radius
        #of measurement channels for calculation of chiPert
        self.tPulse = np.zeros([self.numChan])
        self.aPulse = np.zeros([self.numChan])
        self.rPulse = np.zeros([self.numChan])
        self.teBackground = np.zeros([self.numChan])


        indexStart = (np.abs(self.pulseTime
                                  - self.eleTempTime)).argmin()

        indexEnd = (np.abs((self.pulseTime+self.timeRange)
                                  - self.eleTempTime)).argmin()

        #temporary variables for storage
        tPeak = np.zeros(self.numChan)
        amp = np.zeros(self.numChan)
        rad = np.zeros(self.numChan)
        teBackground = np.zeros(self.numChan)

        #loop through channels
        for i in range(self.numChan):

            #Look at temperatures during time period of interest
            selectedTe = self.eleTemp[i,indexStart:indexEnd]
            selectedgradTe = self.gradTemp[i,indexStart:indexEnd]
                
            #The peak temperature
            peakTe = np.amax(selectedTe)

            #The background temperature at start of time period of interest
            backgroundTe = selectedTe[0]

            #index of peak temperature
            peakIndex = np.argmax(selectedTe)

            #time for peak temperature
            peakTime = self.eleTempTime[i,peakIndex+indexStart]

            #this is where the plotting routine was before

            #Find the index for the radius
            radiusIndex = (np.abs(self.radiusDiagTime[i,:]
                                  - peakTime)).argmin()

            #look up the radius of a given channel
            #Use radius mapped to magnetic axis
            radius = self.radiusDiag[i,radiusIndex]

            #store calculated peaks in temporary variables
            tPeak[i] = peakTime
            amp[i] = peakTe - backgroundTe
            rad[i] = radius
            teBackground[i] = backgroundTe
        
            #Store peaks in final variables
            self.tPulse = tPeak
            self.aPulse = amp
            self.rPulse = rad
            self.teBackground = teBackground

        print('    Data Prepared')

        return

    def calcChiPert(self): 

        ######## Pulse propagation inverse speed calculation######

        popt, pcov = curve_fit(linear, self.rPulse, self.tPulse)
        tSlope = popt[0]
        tSlope_err = pcov[0,0]**0.5

        if self.plotPulse: 
            plt.ion()
            fig, ax = plt.subplots()
            plt.scatter(self.rPulse, self.tPulse) 
            plt.plot(self.rPulse, popt[1] + tSlope * self.rPulse)
            plt.xlabel('Radius')
            plt.ylabel('Time of max Te')


        #Uncorrected pulse velocity
        vPdag = 1/tSlope
        print('vPdag: ' + str(vPdag))

        #Correct for curvature and Shafranov shift
        vP = np.sqrt(self.elong)*((self.a)/(self.a-self.s))*vPdag
        print('vP: ' + str(vP))

        #Pulse veloctiy error
        vP_err = np.sqrt(vP**2 * vPdag**2 * tSlope_err**2)
        print('vP_err: ' + str(vP_err))


        ######## Pulse amplitude decay rate calculation##########
        logApulse = np.log10(self.aPulse*1000)

        popt, pcov = curve_fit(linear, self.rPulse,logApulse)
        aSlope = popt[0]
        aSlope_err = pcov[0,0]**0.5

        # Time index to start and end amplitude decay rate calulcation
        indexStart = (np.abs(self.pulseTime
                                  - self.eleTempTime)).argmin()

        indexEnd = (np.abs((self.pulseTime+self.timeRange)
                                  - self.eleTempTime)).argmin()

        if self.plotPulse:
            plt.ion()
            fig, ax = plt.subplots()
            plt.scatter(self.rPulse, logApulse) 
            plt.plot(self.rPulse, popt[1] + aSlope * self.rPulse)

            plt.xlabel('Radius')
            plt.ylabel('logApulse')


            plt.ion()
            fig, ax = plt.subplots()


        for i in range(self.numChan):

            #Look at temperatures during time period of interest
            selectedTe = self.eleTemp[i,indexStart:indexEnd]
            selectedgradTe = self.gradTemp[i,indexStart:indexEnd]
                
            #The peak temperature
            peakTe = np.amax(selectedTe)

            #index of peak temperature
            peakIndex = np.argmax(selectedTe)

            #time for peak temperature
            peakTime = self.eleTempTime[i,peakIndex+indexStart]

            #Plot pulses if desired
            if self.plotPulse:
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']    

                plt.plot(self.eleTempTime[i,indexStart:indexEnd]*1000,
                    (self.eleTemp[i,indexStart:indexEnd]), #-backgroundTe
                    color = colors[i%7],linewidth=2, label = str(self.rhosC[i])[0:4])
                
                plt.scatter(self.tPulse*1E3, self.aPulse + self.teBackground, label = '_nolegend_', c='k')


                axes = plt.gca()
                axes.set_title(str(self.pulseTime))
                axes.set_ylabel('Pulse Amplitude (keV)',fontsize=20)
                axes.set_xlabel('Time (ms)',fontsize=20)

        
        if self.plotPulse:
            plt.legend(fontsize=16, title = r'$\rho_{tor}$', loc='right')


        #Calculated uncorrected alpha radial damping parameter (in dB)
        alphaDag = np.abs(10*self.a*aSlope)

        #Correct for Shafranov Shift
        alpha = ((self.a-self.s)/self.a)*alphaDag
        print('alpha: ' + str(alpha))

        # calulcate error on alpha calculation
        alpha_err = np.sqrt(100 * (self.a-self.s)**2 * aSlope_err**2)
        print('alpha_err: ' + str(alpha_err))

        #Calculate the perturbative thermal diffusivity
        const = 4.2
        chiPert = const * self.ac * vP /alpha
        print('chiPert: ' + str(chiPert))

        #Calculate the error on the perturbative thermal diffusivity
        const_err = 0.1 * const #10% error according to Creely's thesis
        chiPert_err = (self.ac * vP / alpha)**2 * const_err**2 +\
                        (const * self.ac / alpha)**2 * vP_err**2 +\
                        (const * self.ac * vP / alpha**2)**2 * alpha_err**2 

        print('chiPert_err: ' + str(chiPert_err))


        return chiPert , chiPert_err