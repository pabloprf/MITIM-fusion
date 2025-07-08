'''
From NTH
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from gacodefuncs import *
import statsmodels.api as sm
from cgyro.data import cgyrodata
import math
#from omfit_classes import omfit_gapy

def grab_cgyro_nth(data_dir, tstart_t, plotflag, printflag, file = None):
    
    #Get all the relevant simulation quantities
    data = cgyrodata(data_dir+'/')
    print(dir(data))
    data.getflux()
    rho_star=(data.vth_norm/((1.602e-19*data.b_unit)/(data.mass_norm*1e-27)))/data.a_meters
    nt=data.n_time
    t=data.t
    dt=t[0]
    ky=abs(data.ky)
    n_n=data.n_n
    n_spec=data.n_species
    n_field=data.n_field
    #print(n_field)
    flux = data.ky_flux
    #Shape should be (species,flux,field,ky,time)
    #rint(dir(data))
    roa=data.rmin
    #print data.__init__
    tmax=np.amax(t)
    sflux = np.sum(flux,axis=3)
    kflux=np.mean(flux,axis=4)
    tkflux=np.sum(kflux,axis=2)
    #Values of gradients and GB normalizations
    alti=data.dlntdr[0] #technically ion 1 scale length
    alte=data.dlntdr[n_spec-1]
    alne=data.dlnndr[n_spec-1]
    qgb=data.q_gb_norm
    ggb=data.gamma_gb_norm
    pgb=data.pi_gb_norm
    sgb=qgb/data.a_meters

    #Total electron heat flux, sum over field
    eflux_all=sflux[n_spec-1,1,:,:]
    eflux=np.sum(eflux_all,axis=0)
     
    #ES electron heat flux, just electrostatic
    efluxes=eflux_all[0,:]
    
    #EM electron heat flux, just A_||
    efluxem=eflux_all[1,:]
    
    #Total electron particle flux, sum over fields
    epflux_all=sflux[n_spec-1,0,:,:]
    epflux=np.sum(epflux_all,axis=0)

    #Total impurity particle flux, sum over fields (***NOTE THIS IS FOR SPECIES n_Spec -3******)
    impflux_all=sflux[n_spec-3,0,:,:]
    impflux=np.sum(impflux_all,axis=0)

    #Depending on if electrostatic of E&M output components
    if n_field ==1:
        epfluxp=epflux_all[0,:]
        epfluxap=epflux_all[0,:]*0.0 #Zero out the A_par array
        epfluxbp=epflux_all[0,:]*0.0 #Zero out the B_par array

    if n_field == 2:
        epfluxp=epflux_all[0,:]
        epfluxap=epflux_all[1,:]
        epfluxbp=epflux_all[0,:]*0.0 #Zero out the B_par array

    if n_field ==3:
        epfluxp=epflux_all[0,:]
        epfluxap=epflux_all[1,:]
        epfluxbp=epflux_all[2,:]
        
    
    #Total (ES+EM) ion flux summed over ions
    iflux_all=sflux[0:n_spec-1,1,:,:]
    iflux=np.sum(iflux_all,axis=0) #Sum over ions
    iflux=np.sum(iflux,axis=0) #Sum over fields

    #Sum the momentum flux (all species)
    mflux_all=sflux[0:n_spec,2,:,:]
    mflux=np.sum(mflux_all,axis=0) # Sum over species
    mflux=np.sum(mflux,axis=0) # Sum over fields

    #Total electron turbulent exchange, sm over all fields

    #Put in an option if turbulent exchange is not enabled
    if sflux.shape[1] == 4:
        turflux_all=sflux[n_spec-1,3,:,:]
    else:
        turflux_all=sflux[n_spec-1,2,:,:]*0.0 #Fill in with dummy values and 0
        
    turflux=np.sum(turflux_all,axis=0)
        
    
    if np.amax(ky) > 1.0:
        e_ind=np.where(ky > 1.0)[0]
        eflux_elec_tmp1=flux[:,:,:,e_ind,:]
        eflux_elec_tmp2=np.sum(eflux_elec_tmp1,axis=3)
        eflux_elec_all=eflux_elec_tmp2[n_spec-1,1,:,:]
        eflux_elec=np.sum(eflux_elec_all,axis=0)
        
    #Define max values for plot
    imax=np.amax(iflux)*qgb
    emax=np.amax(eflux)*qgb
    smax=imax+emax

    #Determine if the time step has changed mid simulation
    #Change the value of tstart internally if it has
    tend=data.t[nt-1]
    tstart=(np.abs(t - tstart_t)).argmin()
    
    #Take the mean values of the fluxes
    m_qe= np.mean(eflux[int(tstart):int(nt)+1])
    m_qi= np.mean(iflux[int(tstart):int(nt)+1])
    m_ge= np.mean(epflux[int(tstart):int(nt)+1])
    m_qe_elec=np.mean(eflux_elec[int(tstart):int(nt)+1])
    m_ge_p=np.mean(epfluxp[int(tstart):int(nt)+1])
    m_ge_ap=np.mean(epfluxap[int(tstart):int(nt)+1])
    m_ge_bp=np.mean(epfluxbp[int(tstart):int(nt)+1])
    m_gimp=np.mean(impflux[int(tstart):int(nt)+1])
    m_mo=np.mean(mflux[int(tstart):int(nt)+1])
    m_tur=np.mean(turflux[int(tstart):int(nt)+1])
    print(nt)

    #Calculate the standard deviations of the fluxes based on autocorrelation
    i_tmp=iflux[int(tstart):int(nt-1)]
    e_tmp=eflux[int(tstart):int(nt-1)]
    ep_tmp=epflux[int(tstart):int(nt-1)]
    imp_tmp=impflux[int(tstart):int(nt-1)]
    mo_tmp=mflux[int(tstart):int(nt-1)]
    tur_tmp=turflux[int(tstart):int(nt-1)]
    i_acf=sm.tsa.acf(i_tmp)
    i_array=np.asarray(i_acf)
    icor=(np.abs(i_array-0.36)).argmin()
    e_acf=sm.tsa.acf(e_tmp)
    e_array=np.asarray(e_acf)
    ecor=(np.abs(e_array-0.36)).argmin()
    ep_acf=sm.tsa.acf(ep_tmp)
    ep_array=np.asarray(ep_acf)
    epcor=(np.abs(ep_array-0.36)).argmin()
    imp_acf=sm.tsa.acf(imp_tmp)
    imp_array=np.asarray(imp_acf)
    impcor=(np.abs(imp_array-0.36)).argmin()
    mo_acf=sm.tsa.acf(mo_tmp)
    mo_array=np.asarray(mo_acf)
    mocor=(np.abs(mo_array-0.36)).argmin()
    tur_acf=sm.tsa.acf(tur_tmp)
    tur_array=np.asarray(tur_acf)
    turcor=(np.abs(tur_array-0.36)).argmin()

    n_corr_i=(nt-tstart)/(3.0*icor) #Define "sample" as 3 x autocor time
    n_corr_e=(nt-tstart)/(3.0*ecor)
    n_corr_ep=(nt-tstart)/(3.0*epcor)
    n_corr_imp=(nt-tstart)/(3.0*impcor)
    n_corr_mo=(nt-tstart)/(3.0*mocor)
    n_corr_tur=(nt-tstart)/(3.0*turcor)
    print(n_corr_i)
    print(n_corr_e)
    print(n_corr_ep)
    print(n_corr_imp)
    print(n_corr_mo)
    print(n_corr_tur)
    std_qe=np.std(eflux[int(tstart):int(nt-1)])/np.sqrt(n_corr_e)
    std_qi=np.std(iflux[int(tstart):int(nt-1)])/np.sqrt(n_corr_i)
    std_ge=np.std(epflux[int(tstart):int(nt-1)])/np.sqrt(n_corr_ep)
    std_gimp=np.std(impflux[int(tstart):int(nt-1)])/np.sqrt(n_corr_imp)
    std_mo=np.std(mflux[int(tstart):int(nt-1)])/np.sqrt(n_corr_mo)
    std_tur=np.std(turflux[int(tstart):int(nt-1)])/np.sqrt(n_corr_tur)
    
    m_qees= np.mean(efluxes[int(tstart):int(nt-1)])
    m_qeem= np.mean(efluxem[int(tstart):int(nt-1)])

    #Make some string values so they can be truncated
    s_alti=str(alti)
    s_alte=str(alte)
    s_alne=str(alne)
    
    #Print all the simulation information
    print('')
    print('========================================')
    print('Time to start is:')
    print(tstart)
    print('Max simulation time is')
    print(nt)
    print('Simulation Gradients')
    print('a/LTi = '+s_alti[:6])
    print('a/LTe = '+s_alte[:6])
    print('a/Lne = '+s_alne[:6])
    print('')
    print('======================')   
    print('Heat Flux')
    print('======================')   
    print('Q_gb is')
    print(f"{qgb:.4f}")
    print('Mean Qi (in GB)')
    print(f"{m_qi:.4f}")
    print('Qi Std deviation (in GB)')
    print(f"{std_qi:.4f}")
    print('Mean in MW/m^2')
    print(f"{m_qi*qgb:.4f}")
    print('----------------------')
    print('Mean Qe (in GB)')
    print(f"{m_qe:.4f}")
    print('Qe Std deviation (in GB)')
    print(f"{std_qe:.4f}")
    print('Mean in MW/m^2')
    print(f"{m_qe*qgb:.4f}")
    print('Qi/Qe')
    print(f"{m_qi/m_qe:.4f}")
    print('High-k Qe (GB)')
    print(f"{m_qe_elec:.4f}")
    print('')
    print('======================')   
    print('Particle Flux')
    print('======================')
    print('Gamma_gb is:')
    print(f"{ggb:.4f}")
    print('Mean Gamma_e (in GB)')
    print(f"{m_ge:.4f}")
    print('Gamma_e Std deviation (in GB)')
    print(f"{std_ge:.4f}")
    print('Mean in e19/m^2*s')
    print(f"{m_ge*ggb:.4f}")
    print('Mean Impurity Flux (GB)')
    print(f"{m_gimp:.4e}")
    print('Gamma_imp Std. deviation (in GB)')
    print(f"{std_gimp:.4e}")
    print('Electron Particle Flux Components(GB)')
    print('Phi: 'f"{m_ge_p:.4f}")
    print('A_||: 'f"{m_ge_ap:.4f}")
    print('B_||: 'f"{m_ge_bp:.4f}")
    print('')
    print('======================')   
    print('Momentum Flux')
    print('======================')
    print('Pi_gb is:')
    print(f"{pgb:.4f}")
    print('Mean Pi (in GB)')
    print(f"{m_mo:.4f}")
    print('Momentum Flux Std deviation (in GB)')
    print(f"{std_mo:.4f}")
    print('Mean in J/m^2')
    print(f"{m_mo*pgb:.4f}")
    print('')
    print('======================')   
    print('Turbulent Exchange')
    print('======================')
    print('S_gb is:')
    print(f"{sgb:.4f}")
    print('Mean Elec Turb Ex. (in GB)')
    print(f"{m_tur:.4f}")
    print('Turb Ex. Std deviation (in GB)')
    print(f"{std_tur:.4f}")

    #Print the results to results file
    if printflag ==1:
       file1=open(file,"a")
       file1.write('r/a = {0:4f} \t a/Lne = {1:4f} \t a/LTi = {2:4f} \t a/LTe = {3:4f} \t Qi = {4:4f} \t Qi_std = {5:4f} \t Qe = {6:4f} \t Qe_std = {7:4f} \t Ge = {8:4f} \t Ge_std = {9:4f} \t Gimp = {10:.4e} \t Gimp_std = {11:.4e} \t Pi = {12:4f} \t Pi_std = {13:4f} \t S = {14:4f} \t S_std = {15:4f} \t Q_gb = {16:4f} \t G_gb = {17:4f} \t Pi_gb = {18:4f} \t S_gb = {19:4f} \t tstart = {20:4f} \t tend = {21:4f} \n'.format(roa,alne,alti,alte,m_qi,std_qi,m_qe,std_qe,m_ge,std_ge,m_gimp,std_gimp,m_mo,std_mo,m_tur,std_tur,qgb,ggb,pgb,sgb,tstart,nt))
       file1.close()
          


    #Plot the results               
    if plotflag == 1:

        summax=2.0*(m_qe*qgb+m_qi*qgb)
        maxf=np.amax([np.amax(iflux)*qgb,np.amax(eflux)*qgb])
        pmax=np.amin([2.0*(m_qe*qgb+m_qi*qgb),maxf])
        gmax=np.amax(epflux*ggb)
        gmin=np.amin(epflux*ggb)
        
        # Setting the style for the plots
        plt.rc('axes', labelsize=25)
        plt.rc('xtick', labelsize=25)
        plt.rc('ytick', labelsize=25)

        plt.rcParams['font.family'] = 'serif'

        # Create a figure with 3 subplots vertically stacked (3 rows, 1 column)
        fig, ax = plt.subplots(2, 1, figsize=(10.0, 9.0))

        # First plot (Q vs Time)
        ax[0].plot(t, iflux * qgb, 'b-')
        ax[0].plot([tstart, tend], [m_qi * qgb, m_qi * qgb], 'b-', linewidth=4.0)
        ax[0].plot([tstart, tend], [(m_qi + std_qi) * qgb, (m_qi + std_qi) * qgb], 'b-', linewidth=2.0)
        ax[0].plot([tstart, tend], [(m_qi - std_qi) * qgb, (m_qi - std_qi) * qgb], 'b-', linewidth=2.0)
        ax[0].set_title('Q$_e$ and Q$_i$ vs Time',fontsize=18)
        ax[0].set_xlabel('a/c$_s$',fontsize=15)
        ax[0].set_ylabel('MW/m$^2$',fontsize=15)
        ax[0].axis([0.0, tmax, 0.0, pmax])

        # Second plot (Qe vs Time)
        ax[0].plot(t, eflux * qgb, 'r-')
        ax[0].plot([tstart, tend], [m_qe * qgb, m_qe * qgb], 'r-', linewidth=4.0)
        ax[0].plot([tstart, tend], [(m_qe + std_qe) * qgb, (m_qe + std_qe) * qgb], 'r-', linewidth=2.0)
        ax[0].plot([tstart, tend], [(m_qe - std_qe) * qgb, (m_qe - std_qe) * qgb], 'r-', linewidth=2.0)

        ax[0].fill_between([tstart,tend], [(m_qe - std_qe) * qgb, (m_qe - std_qe) * qgb], [(m_qe + std_qe) * qgb, (m_qe + std_qe) * qgb], color='red', alpha=0.2)
        ax[0].fill_between([tstart,tend], [(m_qi - std_qi) * qgb, (m_qi - std_qi) * qgb], [(m_qi + std_qi) * qgb, (m_qi + std_qi) * qgb], color='blue', alpha=0.2)
        
        #ax[0].plot(t, eflux * qgb, 'g-')  # You can customize this line further
        ax[0].tick_params(axis='x', labelsize=12)  # x-axis tick labels font size reduced
        ax[0].tick_params(axis='y', labelsize=12)
        factor=10 ** 3
        factor2=10 ** 2
        num=math.floor(m_qe*qgb*factor)/factor
        num2=math.floor(std_qe*qgb*factor)/factor
        ax[0].text(10,pmax*0.8,'$Q_e$='+str(num)+' +/- '+str(num2),fontsize=18,color='red')

        num=math.floor(m_qi*qgb*factor)/factor
        num2=math.floor(std_qi*qgb*factor)/factor
        num3=math.floor(m_qi/m_qe*factor2)/factor2
        ax[0].text(10,pmax*0.88,'$Q_i$='+str(num)+' +/- '+str(num2),fontsize=18,color='blue')
        ax[0].text(10,pmax*0.72,'$Q_i/Q_e$='+str(num3),fontsize=18)
        
        # Third plot (epflux vs Time)
        ax[1].plot(t, epflux * ggb, 'g-')  # Plotting epflux on the third subplot
        ax[1].set_title('$\Gamma_e$ vs Time',fontsize=18)
        ax[1].set_xlabel('a/c$_s$',fontsize=15)
        ax[1].set_ylabel('1e19/m$^2$s',fontsize=15)
        ax[1].tick_params(axis='x', labelsize=12)  # x-axis tick labels font size reduced
        ax[1].plot([tstart, tend], [m_ge * ggb, m_ge * ggb], 'g-', linewidth=4.0)
        ax[1].plot([tstart, tend], [(m_ge + std_ge) * ggb, (m_ge + std_ge) * ggb], 'g-', linewidth=2.0)
        ax[1].plot([tstart, tend], [(m_ge - std_ge) * ggb, (m_ge - std_ge) * ggb], 'g-', linewidth=2.0)
        ax[1].fill_between([tstart,tend], [(m_ge - std_ge) * ggb, (m_ge - std_ge) * ggb], [(m_ge + std_ge) * ggb, (m_ge + std_ge) * ggb], color='green', alpha=0.2)

        ax[1].plot(t, epflux * ggb, 'g-')
        ax[1].plot(t,t*0.0,'k',linestyle='--')
        ax[1].tick_params(axis='y', labelsize=12)
        ax[1].axis([0.0, tmax, gmin, gmax])

        factor=10 ** 3
        num=math.floor(m_ge*ggb*factor)/factor
        num2=math.floor(std_ge*ggb*factor)/factor
        ax[1].text(10,gmax*0.8,'$\Gamma_e$='+str(num)+' +/- '+str(num2),fontsize=18)
        ax[1].text(10,gmin*0.95,str(int(tend))+' a/c$_s$',fontsize=18)
        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    return roa,alne,alti,alte,m_qi,std_qi,m_qe,std_qe,m_ge,std_ge,m_gimp,std_gimp,m_mo,std_mo,m_tur,std_tur,qgb,ggb,pgb,sgb,tstart,nt


if __name__ == "__main__":
    # Example usage
    data_dir = sys.argv[1]  # Directory containing the cgyro data
    tstart = float(sys.argv[2])  # Start time for analysis
    plotflag = int(sys.argv[3])  # Flag to indicate if plotting is required
    printflag = int(sys.argv[4])  # Flag to indicate if printing results is required
    file = sys.argv[5] # Print to this file

    grab_cgyro_nth(data_dir, tstart, plotflag, printflag, file = file)

