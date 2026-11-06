import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent/"Data"
FILEPATH = DATA_DIR/"res6978__bias5V_powersweep_Dec4.csv"
DATAPATH = DATA_DIR/"5umGapRes_6789MHz_data"
# -------------------------------------------
'''Import E field data from 2D HFSS simulation
'''

debye=3.34e-30
Ae=1.6e-19*1e-10
hbar=1.05e-34
length=187.5e-6*4*18
Edc_EnergyInJ_2dSim=5.8015E-011
Edc_EnergyInJ_2dSim_Total=Edc_EnergyInJ_2dSim*length

Capacitance=Edc_EnergyInJ_2dSim_Total*2
Voltage_at_1Photon=np.sqrt(hbar*2*np.pi*6.789e9*2/Capacitance)   



Es_MS=np.loadtxt(DATAPATH/"LS5LW5_MS.fld",skiprows=1)
Es_MS=Es_MS[:,3]
Es_MS=Es_MS[Es_MS!=0]
Es_MSenergy=0.5*11*8.85e-12*Es_MS**2 
vfrac_MS=np.sum(Es_MSenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=Es_MS*1.0
Es_energy=Es_MSenergy*1.0

Es_MAt=np.loadtxt(DATAPATH/"LS5LW5_MAt.fld",skiprows=1)
Es_MAt=Es_MAt[:,3]/10
Es_MAt=Es_MAt[Es_MAt!=0]
Es_MAtenergy=0.5*10*8.85e-12*Es_MAt**2  
vfrac_MAt=np.sum(Es_MAtenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,Es_MAt))
Es_energy=np.concatenate((Es_energy,Es_MAtenergy))

Es_MAs=np.loadtxt(DATAPATH/"LS5LW5_MAs.fld",skiprows=1)
Es_MAs=Es_MAs[:,3]/10
Es_MAs=Es_MAs[Es_MAs!=0]
Es_MAsenergy=0.5*10*8.85e-12*Es_MAs**2 # x1 (air) instead of 10 ? 
vfrac_MAs=np.sum(Es_MAsenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,Es_MAs))
Es_energy=np.concatenate((Es_energy,Es_MAsenergy))

Es_SA_perp=np.loadtxt(DATAPATH/"LS5LW5_SA_perp.fld",skiprows=1)
Es_SA_perp=Es_SA_perp[:,3]
Es_SA_perp=Es_SA_perp[Es_SA_perp!=0]
Es_SA_perpenergy=0.5*11*8.85e-12*Es_SA_perp**2  
vfrac_SA_perp=np.sum(Es_SA_perpenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total

Es_SA_para=np.loadtxt(DATAPATH/"LS5LW5_SA_para.fld",skiprows=1)
Es_SA_para=Es_SA_para[:,3]
Es_SA_para=Es_SA_para[Es_SA_para!=0]
Es_SA_paraenergy=0.5*11*8.85e-12*Es_SA_para**2
vfrac_SA_para=np.sum(Es_SA_paraenergy)*1e-9*1e-9*length/Edc_EnergyInJ_2dSim_Total
Es=np.concatenate((Es,np.sqrt(Es_SA_para**2+Es_SA_perp**2)))
Es_energy=np.concatenate((Es_energy,Es_SA_paraenergy+Es_SA_perpenergy))

# -------------------------------------------
E_ave = np.mean(np.abs(Es))
V = 5
print(E_ave)
df = pd.read_csv(FILEPATH)

col_n = 'Photon Occupation'
col_power = 'Power'
col_loss = 'Loss Tangent'
col_shift = 'Resonator Frequency'
col_bias = 'Voltage Frequency'

df['dE_dt'] = 2 * V * df[col_bias] * E_ave / np.sqrt(3)
df['Eac_sq'] = df[col_n] * (Voltage_at_1Photon**2) * (E_ave**2)

df_sorted = df.sort_values(by=col_power)

df_out = pd.DataFrame({
            "Sweep Rate": df_sorted['dE_dt']*1e-6,
            "Loss": df_sorted['Loss Tangent'],
            "Drive Power": df_sorted['Power']
        }).dropna()

df_out.to_csv(SCRIPT_DIR/"loss_sweep.csv", index=False)

unique_powers = df_sorted[col_power].unique()

colors = [cm.copper(x) for x in np.linspace(0, 1, len(unique_powers))]

plt.figure(figsize=[14, 10])

for i, power in enumerate(unique_powers):
    subset = df_sorted[df_sorted[col_power] == power]

    plt.scatter(subset['dE_dt']*1e-6, subset[col_loss]*1e6,
                color=colors[i], label=f'{power}')

    plt.axhline(7.8, color='k', linestyle='--')

    plt.xscale('log')
    plt.xlabel(r'$\dot{E}_{DC}$ (V/$\mu$m s)', fontsize=20)
    plt.ylabel(r'$\tan\delta$ (x $10^6$)', fontsize=20)
    plt.title('TLS Loss Tangent vs. Bias Sweep Rate', fontsize=24)

    plt.legend(title='Drive Power (dBm)', bbox_to_anchor=(1.22, 1), loc='upper right', fontsize=12, title_fontsize=14)
    plt.tick_params(labelsize=18)

plt.tight_layout()
plt.show()
