# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:32:16 2022

@author: tennant3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import csv
from matplotlib import cm

'''local libraries
'''

import Labber_Log_Resonators as llr
import fit_fun as fit

#%%

'''load losstan/frequencies of resonator measurements
'''

r,a, biasfreq=llr.load("Yaniv/Data/New/6978MHzRes_5Vbias/share/res6978GHz_5V_full.pickle")

#%%

'''assign bias frequencies
'''

bias = np.array([100, 215, 464, 1000, 2154, 4642, 10000, 21544, 46416, 100000])

biasf = np.array([])

for i in range(10):
    for j in range(15):
        biasf = np.append(biasf, bias[i]) 

'''assign variable values
'''
        
drive_pow=a[2].reshape((-1,15))
freqdivn=(biasf/a[0]).reshape((-1,15))
losstan=a[1].reshape((-1,15))
freq=a[4].reshape((-1,15))
numph=a[0].reshape((-1,15))
bf=biasf.reshape((-1,15))

evenly_spaced_interval = np.linspace(0, 1, 15)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure()
#plt.clf()
for i in range(drive_pow[0,:].size):
    
    plt.scatter(freqdivn[:,i],losstan[:,i]*1e6, color=colors[i], label='{}'.format(drive_pow[0,i]))#,c=(np.log10(biasfreq)-np.log10(biasfreq.min()))/(np.log10(biasfreq).max()-np.log10(biasfreq).min()))
    plt.xscale('log')
    plt.xlabel('bias freq per photon number')
    plt.ylabel('Losstan ( x 1e6)')
    plt.legend()
    plt.title('6.978 GHz 5V Bias Reduced Receiver Attn.')

# #%%

# '''define TLS model for fitting
# '''

# import scipy.optimize
# def tand(ksi):
#     import scipy.integrate
#     def func(x,y,ksi):
#         integral=y/x*(1-np.exp(-x**2*y/(ksi*np.sqrt(1-x**2))))
#         return integral
#     def func2(x,ksi):
#         integral=(0.5+(ksi*np.sqrt(1-x**2)/x**2+ksi**2*(1-x**2)/x**4)*\
#                   np.exp(-x**2/ksi/np.sqrt(1-x**2))-ksi**2*(1-x**2)/x**4)/x
#         return integral
#     loss=3*ksi*scipy.integrate.dblquad(\
#             func,0.0,1.0,lambda x: 0,lambda x: 1,args=[ksi],epsabs=1e-15,epsrel=1e-15)[0]
# #    loss=3*ksi*scipy.integrate.quad(func2,0.0,1.0,args=(ksi),epsabs=1e-15,epsrel=1e-15,limit=100)[0]
#     return loss
# def fitLL(x,a,b):
#     y=a*x+b
#     return y

# ksi_t=np.logspace(-10,2,50,dtype=np.longdouble)
# tand_t0=np.array([tand(k) for k in ksi_t])
# print('\a')
# plt.figure(1)
# plt.clf()
# plt.loglog(ksi_t,tand_t0)

# ind=ksi_t<1e-2
# xLL=scipy.optimize.curve_fit(fitLL,np.log10(ksi_t[ind]),np.log10(tand_t0[ind]))
# plt.loglog(ksi_t,10**(xLL[0][1])*ksi_t**(xLL[0][0]))
# class tand_class:
#     def __init__(self,xLL,ksi_t,tand_t0):
#         self.a=xLL[0][0]
#         self.b=xLL[0][1]
#         self.ksim=ksi_t
#         self.tandm=tand_t0
#     def tand(self,ksis):
#         ksis=np.array(ksis)
#         t=[]
#         for ksi in ksis:
#             if ksi<1e-2:
#                 t.append(10**self.b*ksi**self.a)
#             elif ksi<1e2:
#                 t.append(np.interp(ksi,self.ksim,self.tandm))
#             else:
#                 t.append(1)
#         return np.array(t)
# tandi=tand_class(xLL,ksi_t,tand_t0)
# plt.loglog(ksi_t,tandi.tand(ksi_t),'-',ksi_t,tand_t0,'.')
# print('\a')    

#%%

'''With imported E field values, calculate surface participation ratio 
'''

debye=3.34e-30
Ae=1.6e-19*1e-10
gap=False
hbar=1.05e-34
filename_add='Yaniv/Data/New/6978MHzRes_5Vbias/share/LS5LW5_'

tandh=2.2e-6
tand0=21.5e-6
p=3*debye
noise=300
E2_at_1Joule=161109638071.587**2
Edc_per_V=88068.2781913607
vfrac=100911.414042574*3e-9
file='..\\..\\E_Field\\LS5LW5_'

length=187.5e-6*4*18
Edc_EnergyInJ_2dSim=5.80006192728241E-011
Edc_EnergyInJ_2dSim_Total=Edc_EnergyInJ_2dSim*length
Capacitance=Edc_EnergyInJ_2dSim_Total*2
Voltage_at_1Photon=np.sqrt(hbar*2*np.pi*6.338e9*2/Capacitance)   
varfile='6338'



Es_MS=np.loadtxt(filename_add+'MS'+'.fld',skiprows=1)
Es_MS=Es_MS[:,3]
Es_MS=Es_MS[Es_MS!=0]
Es_MSenergy=0.5*11*8.85e-12*Es_MS**2  # possible issue should be 11^2/10 ? 11 - Si, 10 - AlO
vfrac_MS=np.sum(Es_MSenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=Es_MS*1.0
Es_energy=Es_MSenergy*1.0

Es_MAt=np.loadtxt(filename_add+'MAt'+'.fld',skiprows=1)
Es_MAt=Es_MAt[:,3]/10
Es_MAt=Es_MAt[Es_MAt!=0]
Es_MAtenergy=0.5*10*8.85e-12*Es_MAt**2   # x1 (air) instead of 10 ? this is correct
vfrac_MAt=np.sum(Es_MAtenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,Es_MAt))
Es_energy=np.concatenate((Es_energy,Es_MAtenergy))

Es_MAs=np.loadtxt(filename_add+'MAs'+'.fld',skiprows=1)
Es_MAs=Es_MAs[:,3]/10
Es_MAs=Es_MAs[Es_MAs!=0]
Es_MAsenergy=0.5*10*8.85e-12*Es_MAs**2 # x1 (air) instead of 10 ? 
vfrac_MAs=np.sum(Es_MAsenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,Es_MAs))
Es_energy=np.concatenate((Es_energy,Es_MAsenergy))

Es_SA_perp=np.loadtxt(filename_add+'SA_perp'+'.fld',skiprows=1)
Es_SA_perp=Es_SA_perp[:,3]
Es_SA_perp=Es_SA_perp[Es_SA_perp!=0]
Es_SA_perpenergy=0.5*11*8.85e-12*Es_SA_perp**2   # one of these should have an inverse \epsilon_sa ?
vfrac_SA_perp=np.sum(Es_SA_perpenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total


Es_SA_para=np.loadtxt(filename_add+'SA_para'+'.fld',skiprows=1)
Es_SA_para=Es_SA_para[:,3]
Es_SA_para=Es_SA_para[Es_SA_para!=0]
Es_SA_paraenergy=0.5*11*8.85e-12*Es_SA_para**2
vfrac_SA_para=np.sum(Es_SA_paraenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,np.sqrt(Es_SA_para**2+Es_SA_perp**2)))
Es_energy=np.concatenate((Es_energy,Es_SA_paraenergy+Es_SA_paraenergy))

vfrac=vfrac_MS+vfrac_MAt+vfrac_MAs+vfrac_SA_perp+vfrac_SA_para
ksi_t=np.logspace(-10,10,100)
tand_t=ksi_t*1

# for i in range(len(ksi_t)):
    
#     tand_i=tandi.tand(ksi_t[i]*2*2/2/\
#                       (np.pi*np.abs(Es)*debye*Voltage_at_1Photon**2/hbar))
#     tand_t[i]=sum(tand_i*Es_energy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
   
# tand_t=tand_t/vfrac
        
# n=a[0]*1

# meas_params=(n,a[1],biasfreq,r)
# theory_params=(ksi_t,tand_t,\
#                vfrac,vfrac_MS,vfrac_MAt,vfrac_MAs,vfrac_SA_perp,vfrac_SA_para)
# adjustable_params=(tandh,tand0,p,noise)

# print('\a')

# #%%

# '''fit losstan to TLS model
# '''

# savefig=False

# n,tan_meas,biasfreq,r=meas_params
# ind=biasfreq<3e4
# n,tan_meas,biasfreq,r=n[ind],tan_meas[ind],biasfreq[ind],[r[i] for i in range(len(r)) if ind[i]]
# meas_params=(n,tan_meas,biasfreq,r)
# ksi_t,tand_t,vfrac,vfrac_MS,vfrac_MAt,vfrac_MAs,vfrac_SA_perp,vfrac_SA_para\
#                =theory_params

# adjustable_params=np.array(adjustable_params)
# adjustable_params[2]/=debye
               
# weight=np.array([np.sqrt(a.pcov[1,1]) for a in r])*tan_meas**2

# #tandh,tand0,p,noise=adjustable_params

# def fitme(meas_params=meas_params,p0=np.array(adjustable_params),\
#           theory_params=theory_params,weight=1, bnds=None):
#     n,tan_meas,biasfreq,r=meas_params
#     ksi_t,tand_t,_,_,_,_,_,_=theory_params
#     #factor of two because of 50% symmetry in triangle wave
# #    Vrate=2*(biasfreq+noise)*r[0].Entry['Keysight 33600A Waveform Generator - Voltage CH1']/2
#     #Vrate=2*biasfreq*r[0].Entry['Keysight 33600A Waveform Generator - Voltage CH1']
#     Vrate=2*biasfreq*5
#     '''last number above refers to bias amplitude'''

#     interp = lambda k : np.interp(k,ksi_t,tand_t)
    
#     def lstsqr(x,tandh,tand0,p,noise):
#         n,Vrate=x
#         ksi=(Vrate+noise)/(n*p)
#         tand_theory=interp(ksi)             
#         tand=np.log(tand_theory*tand0+tandh)
#         return tand

#     fit,cov=scipy.optimize.curve_fit(lstsqr,np.matrix([n,Vrate]),np.log(tan_meas)\
#                                      ,p0=p0,bounds=bnds,sigma=weight/tan_meas)
#     print(fit)
#     print(cov)
#     ksi=(Vrate+fit[3])/(n*fit[2])
#     tand=(tan_meas-fit[0])/fit[1]
#     gof=np.sum((np.log(tan_meas)-np.log(interp(ksi)*fit[1]+fit[0]))**2)/len(tan_meas)
#     return fit, ksi, tand, cov, gof


# fit=np.array([1e-8,   1e-7,   5,  400])

# fit, ksi, tand, cov, gof = fitme(p0=fit,bnds=[[1e-10,0.5e-8,0.0001,0.000001],[1e-1,8e-4,50,1e7]],weight=weight)
# #fit, ksi, tand, cov, gof = fitme(p0=fit,bnds=[[1e-10,0.5e-4,.01,0.01],[np.min(tan_meas)-1e-7,2e-1,50,1e7]],weight=weight)
# tandhe,tand0e,pe,noisee=[np.sqrt(cov[k,k]) for k in range(4)]


# #fit=np.array([1e-7,   1e-5,   50,  400])

# #fit, ksi, tand, cov, gof = fitme(p0=fit,bnds=[[9e-9,8e-8,0.01,0.08],[10e-2,20e-2,2000,5000]],weight=weight)
# tandhe,tand0e,pe,noisee=[np.sqrt(cov[k,k]) for k in range(4)]

# tandh,tand0,p,noise=fit

# #%%

# '''Plot and print results
# ''' 

# plt.figure(5)
# plt.clf()
# plt.plot(ksi_t,tand_t)

# plt.scatter(ksi,tand,\
#             c=(np.log10(biasfreq)-np.log10(biasfreq.min()))/(np.log10(biasfreq).max()-np.log10(biasfreq).min()))
# plt.xlabel(r'ksi')
# plt.xlim([1e-6,1e6])
# plt.xscale('log')
# #plt.xlim(min(ksi)*.5,max(ksi)*2)
# #plt.ylim(min(tand)*.5,max(tand)*2)
# #plt.yscale('log')
# plt.ylabel('Normalized Loss tangent')
# cbar=plt.colorbar()
# cbar.ax.set_yticklabels(np.round(biasfreq[::int(len(ksi)/5)],1))
# cbar.set_label('Bias Frequency')



# if savefig:
#     plt.savefig(varfile+'\\Normalized Losstan vs bias rate over photon number.png',bbox='tight')
# print("Dipole moment: %.2e eA"%(p*debye/Ae))
# print("Dipole moment: %.2e Debye"%(p))
# density=tand0/vfrac/np.pi*3*11*8.85e-12/(p*debye)**2
# print("Density: %.2e J^-1m^-3"%density)
# print("Material LossTangent: %e "%(tand0/vfrac))
# print("Volume Fraction Metal Substrate: %f"%(vfrac_MS))
# print("Volume Fraction Metal Air: %f"%(vfrac_MAt+vfrac_MAs))
# print("Volume Fraction Substrate Air: %f"%(vfrac_SA_perp+vfrac_SA_para))
# print(r"High power LossTangent: %e "%(tandh)+chr(177)+" %e"%(tandhe))
# print(r"Low power LossTangent: %e "%(tand0)+chr(177)+" %e"%(tand0e))
# print(r"dipole moment (Debye): %f "%p+chr(177)+" %f"%pe)
# #Vpp=r[0].Entry['Keysight 33600A Waveform Generator - Voltage CH1']
# Vpp=5
# print(r"Noise (V/s): %f "%(noise)+chr(177)+" %f"%(noisee))
# print(r"Goodness of fit: %e"%gof)
# print("covariance matrix:\n"+str(cov))

# --- Added Save Logic ---

# Flatten the 2D arrays into 1D columns
flat_drive = drive_pow.flatten()
flat_numph = numph.flatten()
flat_losstan = losstan.flatten()
flat_freq = freq.flatten()
flat_biasf = bf.flatten()

# Stack them horizontally into a single 2D array
extracted_data = np.column_stack((flat_drive, flat_numph, flat_losstan, flat_freq, flat_biasf))

# Define the CSV headers
csv_header = "drive_power,photon_number,loss_tangent,center_frequency,bias_frequency"

# Save directly to CSV
np.savetxt("Yaniv/Data/New/6978MHzRes_5Vbias/share/resonator_data.csv", extracted_data, delimiter=",", header=csv_header, comments="")
print("Data successfully saved to extracted_resonator_data.csv")
