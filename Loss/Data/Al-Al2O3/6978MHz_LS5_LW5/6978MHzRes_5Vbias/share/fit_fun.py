#!/usr/bin/env python
"""
Fitting functions used to fit resonances

Functions:
---------

s21_model_eric:                 Eric's model
s21_model_yaniv:                Yaniv's model
s21_model_yaniv_for_fitting:    Yaniv's model used for fitting (for convergence)
find_f0:                        Finding f0 and resonance window
fit_convert_y2def:              Convert parameters from y model to default
fit_convert_def2y:              Convert parameters from default to y model
fit_convert_vis2def:            Convert parameters from visually extractable
                                parameters to default
fit_data:                       Main fitting code
iterative_fit:                  Under construction

"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def s21_model_eric(f, f0, Qi, QcR, QcI, s, Tmax):
    """
    S21 fit using a model provided by Eric

    Parameters:
    ----------
    
    f:              frequency array [GHz]
    f0:             resonant frequency [2pi GHz]
    Qi:             internal quality factor
    QcR,QcI:        real, imaginary parts of coupling Q
    s:              slope of linear part
    Tmax:           linear scaling factor
        
    Returns:
    -------

    S21:            transmission coefficient as a function of frequency

    """

    # Create a reduced frequency array
    x = (f - f0)/f0
    # Convert Qc to Qc_hat
    Qch=1/(1/QcR+1/(1j*QcI))
    res =  ( 
            Qch*(-1.j + 2.0*x*Qi)\
            / ( 1.j*Qi - Qch*(-1.j + 2*x*Qi))\
           )
    S21 = s*x + 20.0*np.log10(np.abs(Tmax*res))

    return S21

def s21_model_yaniv(f, f0, Qi, QcR, QcI, s, Tmax):
    """
    S21 fit using a model provided by Yaniv

    Parameters:
    ----------
    
    f:              frequency array [GHz]
    f0:             resonant frequency [2pi GHz]
    Qi:             internal quality factor
    QcR,QcI:        real, imaginary parts of coupling Q
    s:              slope of linear part
    Tmax:           linear scaling factor
        
    Returns:
    -------

    S21:            transmission coefficient as a function of frequency

    """
    f0, A, Q, assym, s, Tmax = fit_convert_def2y((f0, Qi, QcR, QcI, s, Tmax))

    # Create a reduced frequency array
    x = (f - f0)/f0
    res =  1-A*(1+1j*assym)/(1+2j*Q*x)
    S21 = s*x + 20.0*np.log10(Tmax*np.abs(res))

    return S21

def s21_model_yaniv_for_fitting(f, f0, A, Q, assym, s, Tmax):
    """
    S21 fit using a model provided by Yaniv

    Parameters:
    ----------
    
    f:              frequency array [GHz]
    f0:             resonant frequency [2pi GHz]
    A:              magnitude of minimum = Q/Qc
    Q:              Loaded quality factor
    s:              slope of linear part
    Tmax:           linear scaling factor
        
    Returns:
    -------

    S21:            transmission coefficient as a function of frequency

    """

    # Create a reduced frequency array
    x = (f - f0)/f0
    res =  1-A*(1+1j*assym)/(1+2j*Q*x)
    S21 = s*x + 20.0*np.log10(Tmax*np.abs(res))

    return S21

def s21_model_yaniv_for_fitting_Qi(Qi, f, f0, Qc, assym, s, Tmax):
    """
    S21 fit using a model provided by Yaniv

    Parameters:
    ----------
    
    f:              frequency array [GHz]
    f0:             resonant frequency [2pi GHz]
    A:              magnitude of minimum = Q/Qc
    Q:              Loaded quality factor
    s:              slope of linear part
    Tmax:           linear scaling factor
        
    Returns:
    -------

    S21:            transmission coefficient as a function of frequency

    """

    # Create a reduced frequency array
    x = (f - f0)/f0
    res =  1-()*(1+1j*assym)/(1+2j*Q*x)
    S21 = s*x + 20.0*np.log10(Tmax*np.abs(res))

    return S21



def find_f0(raw_data, step=1, show_plot=False, cut=10):
    """
    Calculates the resonance frequency for one frequency-series
    in raw_data

    Parameters:
    ----------

    raw_data:       frequency data for a single temperature, indexed as
                    frequency = data[:,0], I = data[:,1], Q = data[:,2]
    step:           decimation factor for the data
    show_plot:      show a plot of the fit
    cut:            range of 2*cut values centered about zero to consider
                    in finding the minimum frequency

    Returns:
    -------

    (bleft*step, wleft*step,
     ridx*step, ridx*step,
     ridx*step, wright*step,
     bright*step):              left boundary, left window,
                                index of resonance, right window,
                                right boundary
     
    """
    
    data = raw_data[::step]
    x = data[:,0]
    y = 10.0*np.log10(data[:,1]**2 + data[:,2]**2)

    slope = (y[-1]-y[0])/(x[-1]-x[0])
    line_fit = slope*x - slope*x[0]+y[0]
    
    y = y-line_fit
    
    # Calculate lowpass elliptic filter coefficients
    #             order, ripple [dB], stop band attenuation [dB], cutoff
    #             frequency normalized to the Nyquist frequency 
    # Apply the filter to the magnitude of the IQ data
    b,a = signal.ellip(3, 0.1, 100, 0.005, btype="lowpass")
    fgust = signal.filtfilt(b, a, y, method="gust")
    
    dfy = np.diff(fgust)
    dfy = dfy - np.median(dfy)
    
    # Calculate a set of elliptic filter coefficients
    # Apply the filter the derivative the filtered signal
    b,a = signal.ellip(3, 0.1, 100, 0.01,btype="lowpass")
    dfgust = signal.filtfilt(b,a,dfy,method="gust")

    # cut = 10
    
    # Find the frequency where the response is minimized
    fmin = np.argmin(fgust[cut:-cut])+cut
    fmax = np.argmax(fgust[cut:-cut])+cut
    dfmin = np.argmin(dfgust[cut:-cut])+cut
    dfmax = np.argmax(dfgust[cut:-cut])+cut
    
    # First and second derivative checking for minimum finding
    dfdelta = dfmax - dfmin
    
    # Assign the width of the fitting window
    if(dfdelta > 0):
        wleft = dfmin - 2*dfdelta
        wright = dfmax + 2*dfdelta
        if(fmin < dfmax and fmin > dfmin):
            ridx = fmin

        else:
            ridx = int(dfmin + dfdelta/2)
    else:
        wleft = dfmax + 2*dfdelta
        wright = dfmin - 2*dfdelta
        if(fmin > dfmax and fmin < dfmin):
            ridx = fmin
        else:
            ridx = int(dfmin + dfdelta/2)
    
    dx = x.size-ridx
    if(dx > ridx):
        bleft = cut
        bright = ridx*2
    else:
        bleft = ridx-dx
        bright = x.size-cut

    # Check and set the window left and right boundaries
    if(wleft < bleft):
        wleft = bleft
    if(wright > bright):
        wright = bright
        
    # Display the results of the fits
    if(show_plot):
        plt.plot(x[bleft:bright],dfgust[bleft:bright],'b',
                 x[:-1],dfgust,'r',
                 x[ridx],dfgust[ridx],'ro',
                 x[dfmin],dfgust[dfmin],'go',
                 x[dfmax],dfgust[dfmax],'bo')
        plt.axvline(x[wleft],color='g')
        plt.axvline(x[wright],color='g')
        plt.show()

        plt.plot(x[bleft:bright],fgust[bleft:bright],'b',
                 x,fgust,'r',
                 x[dfmin],fgust[dfmin],'bo',
                 x[fmin],fgust[fmin],'go',
                 x[dfmax],fgust[dfmax],'bo',
                 x[ridx],fgust[ridx],'rD')
        plt.axvline(x[wleft],color='g')
        plt.axvline(x[wright],color='g')
        plt.show()
        
        
    return (bleft*step, wleft*step, ridx*step,\
            wright*step, bright*step)
    
def fit_convert_y2def(params,pcov=[]):
    """
    Converts parameters from 'y' model inputs to default inputs
    
    Parameters:
    ----------
    
    params:             Parameters for 'y' model of the form:
                        (f0, A, Q, asym, s, Tmax)
    pcov:               (Optional) Covarient matrix

    Returns:            
    -------

    converted:          Parameters for curve_fit of the form:
                        (f0, Qi, QcR, QcI, s, Tmax)
    
    pcov_convert:       (Optional) If covarient matrix given, returns new
                        covarient matrix.
    """
    A=np.float64(params[1])
    Q=np.float64(params[2])
    asym=np.float64(params[3])
    QcR=Q/A
    QcI=Q/(asym*A)
    Qi=Q/(1-A)
    converted=list(params)
    converted[1]=Qi
    converted[2]=QcR
    converted[3]=QcI
    if len(pcov):
        pcov=np.matrix(pcov)
        #Construct jacobian to get covarient matrix
        J=np.matrix([[1,0,0,0,0,0],\
                    [0,Q/(1-A)**2,1/(1-A),0,0,0],\
                    [0,-Q/A**2,1/A,0,0,0],\
                    [0,-Q/(asym*A**2),1/(asym*A),-Q/(asym**2*A),0,0],\
                    [0,0,0,0,1,0],\
                    [0,0,0,0,0,1]])
        pcov_convert=J*pcov*np.transpose(J)
        return converted, pcov_convert
    else:
        return converted
    
def fit_convert_def2y(params):
    """
    Converts parameters from default inputs to 'y' model inputs
    
    Parameters:
    ----------
    
    params:             Parameters for curve_fit of the form:
                        (f0, Qi, QcR, QcI, s, Tmax)

    Returns:
    -------

    converted:          Parameters for 'y' model of the form:
                        (f0, A, Q, asym, s, Tmax)
    """
    
    Qi=np.float64(params[1])
    QcR=np.float64(params[2])
    QcI=np.float64(params[3])
    Q=1/(1/Qi+1/QcR)
    A=Q/QcR
    assym=QcR/QcI
    converted=list(params)
    converted[1]=A
    converted[2]=Q
    converted[3]=assym
    return converted

def fit_convert_vis2def(params):
    """
    Converts parameters from visually extractable inputs to default model inputs
    
    Parameters:
    ----------
    
    params:             Parameters for visual extraction of the form:
                        (f0, FWHM, min_value, asymmetry, slope, background)

    Returns:
    -------

    converted:          Parameters for curve_fit of the form:
                        (f0, Qi, QcR, QcI, s, Tmax)
    """
    
    f0=params[0]
    Q=f0/params[1]
    Tmax=params[5]
    A=1-params[2]/Tmax    
    converted=fit_convert_y2def((f0, A, Q, params[3], params[4], Tmax))
    
    return converted

    

def fit_data(data, initial_values=[], guess_step=1,model='y'):
    """
    
    Parameters:
    ----------
    
    data:               raw data as expected by find_f0, e.g.
                        frequency = data[:,0]; I = data[:,1]; Q = data[:2];
    initial_values:     initial guesses for curve_fit of the form:
                        (f0, Qi, QcR, QcI, s, Tmax)
    guess_step:         guess for the appropriate decimation factor
    model:              Use the 'y' model or 'e' model to fit the resonator

    Returns:
    -------

    (freq, y0, popt, pcov, initial_values): freq is the frequency array used in the
                                          fit, y0 is the magnitude of the data
                                          squared [dB], popt and pcov are the
                                          optimal coefficients and covariances
                                          returned by curve_fit, and
                                          initial_values are those used by
                                          curve_fit to fit y0 on freq
    
    """
    # Find f0 = argmin(filtered data)
    ixi, f0_min_idx, f0_guess_idx, f0_max_idx, ixf = \
            find_f0(data, show_plot=False, step=guess_step)

    # Read the data into freq as initial frequencies
    # Compute the initial ydB = 10log10(I^2 + Q^2)
    freq = data[:,0][ixi:ixf:guess_step]
    S21 = data[ixi:ixf:guess_step,1] + 1j*data[ixi:ixf:guess_step,2]
    S21mag=np.abs(S21)
    S21dB = 20.0*np.log10(S21mag)

    # Store the estimated angular resonance frequencies
    f0_guess = data[:,0][f0_guess_idx]
    f0_min = data[:,0][f0_min_idx]
    f0_max = data[:,0][f0_max_idx]

    if model=='e':
        if(len(initial_values) != 6):
            # Guess for the internal quality factor
            Qi_guess = 2*(f0_guess/(f0_max-f0_min))
            
            # Guess for the coupling (external) quality factor Qc or Qe
            QcR_guess = Qi_guess
            
            # Guess for QcI, imaginary part of Qc
            QcI_guess = -Qi_guess
        
            # Guess the additive slope parameter
            s_guess = (S21dB[-1]-S21dB[0])/((freq[-1]-f0_guess)/f0_guess - (freq[0]-f0_guess)/f0_guess)
            
            # Guess the constant scaling factor
            Tmax_guess = np.exp(np.log(10)*np.median(S21dB)/10)
            
            initial_values = (f0_guess, Qi_guess, QcR_guess, QcI_guess, s_guess, Tmax_guess) 
        
        sx = (freq-f0_guess)/f0_guess
                        
        sigma = np.ones(len(sx)) - np.exp(-Qi_guess*sx**2)+1e-3
            
            # Check for the correct number of initial parameters
            # Fit the data using Eric's S21 model
       
        popt, pcov = curve_fit(s21_model_eric, freq, S21dB, p0=initial_values,\
                               sigma=sigma, maxfev=10000)
        results = popt
    elif model=='y':
        # Guess the constant scaling factor
        Tmax_guess = np.median(S21mag[:10]+S21mag[-10:])
        Tmax_min = np.min(S21mag)
        Tmax_max = np.max(S21mag)*10
        
        # Guess for the internal quality factor
        Q_guess = f0_guess/(f0_max-f0_min)
        Q_min = freq[0]/(freq[-1]-freq[0])
        Q_max = freq[-1]/(freq[1]-freq[0])
                # Guess for the dip amplitude Q/Qc
        A_guess = 1-np.abs(data[f0_guess_idx,1]+1j*data[f0_guess_idx,2])/Tmax_guess
        A_min = 0
        A_max = 1
        
        # Guess for QcI, imaginary part of Qc
        asym_guess = 0
        asym_min = -np.inf
        asym_max = np.inf
    
        # Guess the additive slope parameter
        s_guess = (S21dB[-1]-S21dB[0])/((freq[-1]-f0_guess)/f0_guess - (freq[0]-f0_guess)/f0_guess)
        s_min = -1e5
        s_max = 1e5
        
        sigma = 1/S21dB        
        if(len(initial_values) != 6):
            initial_values = (f0_guess, A_guess, Q_guess, asym_guess, s_guess, Tmax_guess)
        else:
            initial_values_y = fit_convert_def2y(initial_values)
        popt, pcov = curve_fit(s21_model_yaniv_for_fitting, freq, S21dB, p0=initial_values_y,\
                               sigma=sigma,\
                               bounds=([f0_min, A_min, Q_min, asym_min, s_min, Tmax_min],\
                                         [f0_max, A_max, Q_max, asym_max, s_max, Tmax_max]),\
                               max_nfev=10000)
        results,pcov=fit_convert_y2def(popt,pcov)

    return(results, np.matrix(pcov), initial_values)


def iterative_fit(dbf_dict ):
    """
    Iteratively refines the fit for all temperature data

    Parameters:
    ----------

    dbf_dict:       dB data in frequency
    


    Returns:
    -------

    """

    temp_data = []
    for tkey in sorted(dbf_dict.keys()):
        if(tkey < .8):
            freq_data = []
            for fkey in sorted(dbf_dict[tkey].keys()):
                dbf = dbf_dict[tkey]
                key = fkey
                initial_values = ()
                nval = len(dbf[key])
                mid = int(nval/2)
                print(tkey,fkey)
                freq,S21dB,popt,pcov,guess = fit_data(dbf[key][mid]['data'],\
                                        initial_values,guess_step=1)
                resdict = {}
                resdict.setdefault('freq',[])
                resdict.setdefault('S21dB',[])
                resdict.setdefault('popt',[])
                resdict.setdefault('pcov',[])
                resdict.setdefault('perr',[])
                resdict.setdefault('guess',[])
                for idx in range(nval):
                    ddict = dbf[key][idx]
                    freq,S21dB,popt,pcov,guess = fit_data(ddict['data'],initial_values)
                #    print(guess-popt) 
                    perr = np.sqrt(np.diag(pcov))
                    resdict['freq'].append(freq)
                    resdict['S21dB'].append(S21dB)
                    resdict['popt'].append(popt)
                    resdict['pcov'].append(pcov)
                    resdict['perr'].append(np.sqrt(np.diag(pcov)))
                    resdict['guess'].append(guess)
                    plt.plot(freq,S21dB,'b',alpha=.5)
                    plt.plot(freq,vs21_model_eric(popt,freq,S21dB),'r')
                    plt.plot(freq,vs21_model_eric(guess,freq,S21dB),'g')
                    plt.show()
                freq_data.append([fkey,resdict])
            temp_data.append([tkey,freq_data])

    return (freq_data, temp_data)



