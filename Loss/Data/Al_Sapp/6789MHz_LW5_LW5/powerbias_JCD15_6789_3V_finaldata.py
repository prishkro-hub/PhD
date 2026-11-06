# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:44:25 2024

@author: Daniel
"""


import Labber_Log_Resonators as llr
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import csv
from matplotlib import cm
import fit_fun as fit

#%%

'''import data set and define variables
'''

bias = np.array([100, 231, 534, 1233, 2848, 6579, 15199, 35112, 81113, 187382, 432876, 1e6])

biasf = np.array([])


npow=43
nf=12

for i in range(nf):
    for j in range(npow):
        biasf = np.append(biasf, bias[i])  
        

r,a, biasfreq=llr.load("Yaniv/Data/5umGapRes_6789MHz_data/res6789GHz_3V_full.pickle")

newbias = np.round(np.logspace(2, 5, num=6),0)
drive_pow=a[2].reshape((-1,npow))
freqdivn=(biasf/a[0]).reshape((-1,npow))
losstan=a[1].reshape((-1,npow))
freq=a[4].reshape((-1,npow))
numph=a[0].reshape((-1,npow))
bf=biasf.reshape((-1,npow))

#%%

'''figures
'''

evenly_spaced_interval = np.linspace(0, 1, npow)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure()
#plt.clf()
for i in range(drive_pow[0,:].size):
    
    plt.scatter(freqdivn[:,i],losstan[:,i]*1e6, color=colors[i], label='{}'.format(drive_pow[0,i]))
    plt.xscale('log')
    plt.xlabel('bias freq per photon number')
    plt.ylabel('Losstan ( x 1e6)')
    plt.legend()
    plt.title('6.978 GHz 5V Bias Reduced Receiver Attn.')


evenly_spaced_interval = np.linspace(0, 1, npow)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure()
#plt.clf()
for i in range(drive_pow[0,:].size):
    
    plt.scatter(freqdivn[:,i],freq[:,i]*1e6, color=colors[i], label='{}'.format(drive_pow[0,i]))
    plt.xscale('log')
    plt.xlabel('bias freq per photon number')
    plt.ylabel('Center Frequency ( x 1e6)')
    plt.legend(loc='right')
    #plt.title('6.978 GHz 5V Bias Reduced Receiver Attn.')        


evenly_spaced_interval = np.linspace(0, 1, nf)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure()
#plt.clf()
for i in range(bias.size):
    
    plt.scatter(freqdivn[i,:],losstan[i,:]*1e6, color=colors[i], label='{}'.format(bias[i]))
    plt.xscale('log')
    plt.xlabel('bias freq per photon number')
    plt.ylabel('Losstan ( x 1e6)')
    plt.legend()


evenly_spaced_interval = np.linspace(0, 1, nf)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure()
#plt.clf()
for i in range(bias.size):
    
    plt.scatter(numph[i,:],(freq[i,:]-freq[0,0])*1e-3, color=colors[i], label='{}'.format(bias[i]))
    plt.xscale('log')
    plt.xlabel('photon number',fontsize=20)
    plt.ylabel('Frequency Shift (kHz)',fontsize=20)
    #plt.legend(loc='lower left')
    #plt.title('6.978 GHz 1V Bias Reduced Receiver Attn.')
    #plt.legend(loc='lower left')
    plt.legend(fontsize=10, loc='lower left')
    plt.grid()
    plt.tick_params(labelsize=20)


evenly_spaced_interval = np.linspace(0, 1, nf)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure()
#plt.clf()
for i in range(bias.size):
    
    plt.scatter(numph[i,:],losstan[i,:]*1e6, color=colors[i], label='{}'.format(bias[i]))
    plt.xscale('log')
    plt.xlabel('photon number',fontsize=20)
    plt.ylabel('Losstan ( x 1e6)',fontsize=20)
    plt.legend(loc='lower left')
    plt.legend(fontsize=10)
    plt.grid()
    plt.tick_params(labelsize=20)


#%%

'''Import E field data from 2D HFSS simulation
'''

debye=3.34e-30
Ae=1.6e-19*1e-10
hbar=1.05e-34
filename_add='Yaniv/Data/5umGapRes_6789MHz_data/LS5LW5_'
length=187.5e-6*4*18
Edc_EnergyInJ_2dSim=5.8015E-011
Edc_EnergyInJ_2dSim_Total=Edc_EnergyInJ_2dSim*length

Capacitance=Edc_EnergyInJ_2dSim_Total*2
Voltage_at_1Photon=np.sqrt(hbar*2*np.pi*6.789e9*2/Capacitance)   



Es_MS=np.loadtxt(filename_add+'MS'+'.fld',skiprows=1)
Es_MS=Es_MS[:,3]
Es_MS=Es_MS[Es_MS!=0]
Es_MSenergy=0.5*11*8.85e-12*Es_MS**2 
vfrac_MS=np.sum(Es_MSenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=Es_MS*1.0
Es_energy=Es_MSenergy*1.0

Es_MAt=np.loadtxt(filename_add+'MAt'+'.fld',skiprows=1)
Es_MAt=Es_MAt[:,3]/10
Es_MAt=Es_MAt[Es_MAt!=0]
Es_MAtenergy=0.5*10*8.85e-12*Es_MAt**2  
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
Es_SA_perpenergy=0.5*11*8.85e-12*Es_SA_perp**2  
vfrac_SA_perp=np.sum(Es_SA_perpenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total

Es_SA_para=np.loadtxt(filename_add+'SA_para'+'.fld',skiprows=1)
Es_SA_para=Es_SA_para[:,3]
Es_SA_para=Es_SA_para[Es_SA_para!=0]
Es_SA_paraenergy=0.5*11*8.85e-12*Es_SA_para**2
vfrac_SA_para=np.sum(Es_SA_paraenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,np.sqrt(Es_SA_para**2+Es_SA_perp**2)))
Es_energy=np.concatenate((Es_energy,Es_SA_paraenergy+Es_SA_perpenergy))


#%%

'''figures of e-field distribution
'''

e_ave = round(np.mean(np.abs(Es)))

plt.figure(figsize=[14,12])
plt.hist(np.abs(Es_MS), bins=50, alpha = 0.2, label='MS')
plt.hist(np.abs(Es_MS), bins=50, edgecolor = 'black', fc = 'none')
plt.hist(np.abs(Es_MAt), bins=50, alpha = 0.2, label='MAt')
plt.hist(np.abs(Es_MAt), bins=50, edgecolor = 'black', fc = 'none')
plt.hist(np.abs(Es_MAs), bins=50, alpha = 0.2, label='MAs')
plt.hist(np.abs(Es_MAs), bins=50, edgecolor = 'black', fc = 'none')
plt.hist(np.abs(Es_SA_para), bins=50, alpha = 0.2, label='SA para')
plt.hist(np.abs(Es_SA_para), bins=50, edgecolor = 'black', fc = 'none')
plt.hist(np.abs(Es_SA_perp), bins=50, alpha = 0.2, label='SA perp')
plt.hist(np.abs(Es_SA_perp), bins=50, edgecolor = 'black', fc = 'none')
plt.vlines(np.mean(np.abs(Es)), ymin = 0, ymax = 100000, color = 'red', linewidth = 10, label = f'E ave = {e_ave} V/m')
plt.yscale('log')
plt.xlabel('Electric Field Magnitude (V/m)',fontsize=20)
plt.title('6.789 GHz', fontsize=20)
plt.legend(loc='upper left', fontsize=26)
plt.tick_params(labelsize=20)


mn = [np.mean(np.abs(Es_MS)), np.mean(np.abs(Es_MAt)), np.mean(np.abs(Es_MAs)), np.mean(np.abs(Es_SA_perp)), np.mean(np.abs(Es_SA_para))]
md = [np.median(np.abs(Es_MS)), np.median(np.abs(Es_MAt)), np.median(np.abs(Es_MAs)), np.median(np.abs(Es_SA_perp)), np.median(np.abs(Es_SA_para))]
cats = ['MS', 'MAt', 'MAs', 'SA perp', 'SA para']
plt.figure(figsize=[14,12])
plt.plot(cats, mn, 'o', markersize=20, label='mean')
plt.plot(cats, md, 'o', markersize=20, label='median')
plt.ylabel('Electric Field Magnitude (V/m)',fontsize=20)
plt.xlabel('Resonator Interface',fontsize=20)
plt.title('6.789 GHz', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.tick_params(labelsize=20)


#%%

'''Frequency shift analysis - assume homogeneous e-field taking E-ave from previous cell
'''

V = 3


evenly_spaced_interval = np.linspace(0, 1, bias.size)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure(figsize=[16,14])
for i, b in enumerate(bias):
    dEbias_dt = 2 * V * b * np.mean(np.abs(Es)) / np.sqrt(3)
    Eac_sq = numph[i,:] * (hbar * 2 * np.pi * 6.789e9 * 2 / Capacitance) * np.mean(np.abs(Es))**2
    xaxis = Eac_sq / dEbias_dt
    
    print(xaxis)
    plt.plot(xaxis, (freq[i,:]-freq[0,0])*1e-3, 'o:', color=colors[i], label='{}'.format(b))
    
plt.xscale('log')
plt.xlabel('$E_{AC}^2$ / $\dot{E}_{DC}$ (Vs/m)',fontsize=26)
plt.ylabel('Frequency Shift (kHz)',fontsize=26)
plt.title(f'6.789 GHz {V}V Bias', fontsize=26)
plt.legend(fontsize=20, loc='lower left')
plt.grid()
plt.tick_params(labelsize=20)



evenly_spaced_interval = np.linspace(0, 1, drive_pow[0,:].size)
colors = [cm.copper(x) for x in evenly_spaced_interval]

plt.figure(figsize=[16,14])
for i, d in enumerate(drive_pow[0,:]):
    plt.plot(bias, (freq[:,i]-freq[0,0])*1e-3, 'o:', color=colors[i], label='{}'.format(drive_pow[0,i]))
    
plt.xscale('log')
plt.xlabel('$\dot{E}_{DC}$',fontsize=26)
plt.ylabel('Frequency Shift (kHz)',fontsize=26)
plt.title(f'6.789 GHz {V}V Bias', fontsize=26)
plt.legend(fontsize=10, loc='lower left')
plt.grid()
plt.tick_params(labelsize=20)



dip_mom = []

for i in range(bias.size):
    
    a = np.argmin(freq[i,:]-freq[0,0])
    
    dEbias_dt = 2 * V * bias[i] * np.mean(np.abs(Es)) / np.sqrt(3)
    Eac_sq = numph[i,a] * (hbar * 2 * np.pi *6.789e9 * 2 / Capacitance) * np.mean(np.abs(Es))**2
    xaxis = Eac_sq / dEbias_dt
    print(bias[i], xaxis, (8 * hbar / xaxis)/debye)

    dip_mom = np.append(dip_mom, (8 * hbar / xaxis)/debye)



Ebias = np.array(bias)
Eb = Ebias.tolist()

plt.figure(figsize=[14,12])
plt.plot(Eb, dip_mom, 'o:')
plt.xlabel('Bias Frequency (Hz)',fontsize=26)
plt.ylabel('Dipole Moment (D)',fontsize=26)
plt.title(f'6.789 GHz {V}V Bias', fontsize=26) 
plt.legend(loc='lower left')
plt.grid()
plt.tick_params(axis='both', labelsize=20)
plt.xscale('log')


plt.figure(figsize=[14,12])
plt.plot(Eb[:-5], dip_mom[:-5], 'o:')
plt.xlabel('Bias Frequency (Hz)',fontsize=26)
plt.ylabel('Dipole Moment (D)',fontsize=26)
plt.title(f'6.789 GHz {V}V Bias', fontsize=26) 
plt.legend(loc='lower left')
plt.grid()
plt.tick_params(axis='both', labelsize=20)
plt.xscale('log')

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
np.savetxt("Yaniv/Data/5umGapRes_6789MHz_data/resonator_data_3V.csv", extracted_data, delimiter=",", header=csv_header, comments="")
print("Data successfully saved to extracted_resonator_data.csv")
