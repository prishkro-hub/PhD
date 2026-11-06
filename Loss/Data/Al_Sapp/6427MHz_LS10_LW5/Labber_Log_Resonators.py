# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:19:26 2017

@author: rosen10
Version: 171201

Package to load a set of curves from labber that were made using a VNA or 
other frequency swept method.
Requirements:
    In labber installation there is a folder called Scripts. That folder must
    be in your path variable.
Objects:
    dataset - class containing one trace from the logfile as well as
              information about that trace and useful functions for it
    save - function to save an object using pickle
    load - function to load an object using pickle
    labber_driver_default_names - function returning labber channel names.
    load_data - function that loads a labber logfile and returns a list with
                dataset elements
    losstan - function that plots and returns a numpy array of relevant 
              parameters from a list of datasets.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import Labber
import fit_fun as fit
import scipy.optimize
import copy


class dataset:
    """
    Class containing information about a single VNA trace
    Inputs:
        s21 - complex S21 input (Must be linear!)
        startfreq (np.nan) - The starting frequency of a sweep. If startfreq
                             is nan then function uses centerfreq and span to
                             calculate frequency array
        stopfreq (np.nan) - The last frequency of a sweep
        centerfreq (np.nan) - The center frequency of a sweep
        freq (np.nan) - array of frequencies
        span (np.nan) - The span of a sweep
        delay (0) - The electrical line delay
        averages (1) - Number of averages performed on trace
    Default Variable Outputs:
        self.freq - An array of frequencies for S21
        self.s21 - Complex valued S21 with electrical delay removed
        self.magdB - s21 in units of dB
        self.real - the real part of S21
        self.imag - the imaginary part of S21
        self.averages - The number of averages performed on the trace
    Default Objects:
        self.plot - Plots S21 vs frequency and IQ
        self.fitme - function that fits resonator data
        self.average - function that averages two traces        
    """
    def __init__(self,s21,startfreq=np.nan,stopfreq=np.nan,
                 centerfreq=np.nan,span=np.nan,freq=np.nan,delay=0,averages=1):
        if np.isnan(startfreq):
            startfreq=centerfreq-span/2
            stopfreq=centerfreq+span/2
        if type(freq)==float:
            self.freq=np.linspace(startfreq,stopfreq,len(s21))
        else:
            self.freq=freq
        self.s21=s21*np.exp(1j*2*np.pi*self.freq*delay)
        self.magdB=20*np.log10(np.abs(self.s21))
        self.real=np.real(self.s21)
        self.imag=np.imag(self.s21)
        self.averages=averages
        
    def plot(self, IQ=True, linear=False, real_fit=False, init=False):
        """
        Function that plots S21 vs Freq and I vs Q.
        Inputs:
            IQ (True) - Plots I vs Q in addition to S21 vs Freq
            linear (Flase) - Plots S21 in linear scale or dB
            real_fit (False) - Plots the fitting parameters
            init (False) - Plots the initial guess for fitting parameters. If
                           init is a list of 6 parameters it will use the
                           provided values to plot the guess, otherwise it will
                           pull it from self.
        """
        
        # >>> This is fine for now, but I would consider using something like 
        #     this in the future so that you do not need to keep track of
        #     figure numbers have problems with overlapping figure names 
        #     in the class and there are generally more features with axes than
        #     there are with the standard plt functions
        """
        fig_i, ax_i = plt.subplots()
        fig_i.clf()
        ax_i.plot()
        ax_i.set_xlabel()
        ax_i.set_ylabel()
        ax_i.set_title()
        plt.show() <----- probably need this line to display the figure outside
                          of ipython and it might be worth adding a
                          fig_i.savefig('filename', format='ps, eps, png')

        .
        .
        .

        fig_2, ax_2 = plt.subplots()
        """
        
        plt.figure(1)
        plt.clf()
        if linear==False:
            plt.plot(self.freq,self.magdB,'b.')
            plt.ylabel('S21 (dB)')
        else:
            plt.plot(self.freq,np.abs(self.s21),'b.')
            plt.ylabel('S21 (Linear)')
        if real_fit:
            plt.plot(self.freq,fit.s21_model_yaniv(self.freq,
                                              self.fit_x[0],self.fit_x[1],
                                              self.fit_x[2],self.fit_x[3],
                                              self.fit_x[4],self.fit_x[5]),'r')
            plt.title('Qi = %f'%self.Qi)
        if type(init)==bool:
            if init:
                plt.plot(self.freq,fit.s21_model_yaniv(self.freq,
                                              self.init_x[0],self.init_x[1],
                                              self.init_x[2],self.init_x[3],
                                              self.init_x[4],self.init_x[5]),'g')
        elif len(init)==6:
            plt.plot(self.freq,fit.s21_model_yaniv(self.freq,
                                              init[0],init[1],
                                              init[2],init[3],
                                              init[4],init[5]),'g')
        plt.xlabel('Frequency')
        plt.show()
        if IQ:
            plt.figure(2)
            plt.clf()
            plt.plot(self.real,self.imag,'b.')
            plt.axes().set_aspect('equal', 'datalim')
            plt.xlabel('I')
            plt.ylabel('Q')
            plt.show()
                

    def average(self,new_s21,new_data_averaged=1):
        """
        Function that averages two traces.
        Inputs:
            new_s21 - new trace to be averaged with self
            new_data_averaged (1) - Number of times new trace has been averaged
        Updated Variables:
            self.averages
            self.s21
            self.magdB
            self.real
            self.imag
        """
        self.s21=(self.s21*self.averages+new_s21*new_data_averaged)/(\
                 self.averages+new_data_averaged)
        self.averages=self.averages+new_data_averaged
        self.magdB=20*np.log10(np.abs(self.s21))
        self.real=np.real(self.s21)
        self.imag=np.imag(self.s21)
        
    def fitvis(self,visual):
        """
        Function that uses fitme but passes visually obtainable fit parameters.
        Inputs:
            visual - (f0, FWHM, min_value, asymmetry, slope, background), where
                    f0 is the resonant frequency
                    FWHM is full width half max
                    min_value is the value at the minima
                    asymmetry is the assymetry parameter for the resonator
                    slope is the slope of the linear part
                    background is the value of the background
        """
        init=fit.fit_convert_vis2def(visual)
        self.fitme(init)
        
    def fitme(self,init):
        """
        Function that fits the resonator with an initial set of values.
        If the fit fails it plots the fit with the initial values. If the fit
        succeeds, it plots the initial guess and the fit.
        Inputs:
            init - (f0, Qi, QcR, QcI, s, Tmax), where
                    f0 is the resonant frequency
                    Qi is the internal quality factor
                    QcR is the real coupling quality factor
                    QcI is the complex part of the coupling quality factor
                    s is the slope of the linear part
                    Tmax is the normalization
        Updated Variables:
            self.init_x
            self.init_f0, self.init_Q, self.init_Qc
            self.init_assym, self.init_s, self.init_Tmax
            self.init_Qi
            self.fit_x
            self.f0, self.Q, self.Qc, self.assym, self.s, self.Tmax
            self.Qi
            self.pcov
        """
        
        self.init_x=init
        self.init_f0=init[0]
        self.init_Qi=init[1]
        self.init_Qc=init[2]
        self.init_QcI=init[3]
        self.init_s=init[4]
        self.init_Tmax=init[5]
        self.init_Q=1/(1/init[1]+1/init[2])
        try:
            data=np.transpose(np.array([self.freq,np.real(self.s21),np.imag(self.s21)]))
            x=fit.fit_data(data,init)
            self.fit_x=x[0]
            self.f0=x[0][0]
            self.Qi=x[0][1]
            self.Qc=x[0][2]
            self.QcI=x[0][3]
            self.s=x[0][4]
            self.Tmax=x[0][5]
            self.Q=1/(1/x[0][1]+1/x[0][2])
            self.pcov=x[1]
            self.plot(IQ=False,real_fit=True,init=True)
        except ValueError:
            print("Could not fit to data!!!")
            self.fit_x=[np.nan]
            self.plot(IQ=False,init=init)
            


def save(myobject,filename):
    """
    Saves object to a file
    Inputs:
        myobject - The object to be saved, e.g. list of datasets
        filename - The filename to save to
    """
    pickle.dump(myobject,open(filename, "wb"))
    
def load(filename):
    """
    Loads object from a file saved with 'save' function
    Input:
        filename - The filename to load from
    Output:
        myobject - The object that was loaded
    """
    myobject=pickle.load(open(filename,'rb'))
    return myobject

def labber_driver_default_names(device='N5230C',Entry=None):
    """
    Returns labber driver channel names for specific devices
    Input:
        device ('N5230C') - Device name
    Output:
        DeviceStrings - A dictionary with labber driver channel names
    """
    autofind=False
    if Entry!=None:
        for key in Entry.keys():
            if key[-6:]==' - S21':
                device=key[:-6]
                autofind=True
        
    if device=='N5230C_S12':
        device = 'N5230C'
        
    if autofind==True or device == 'N5230C' or device=='STLSS_VNA_New':
        DeviceStrings=dict()
        DeviceStrings['device']=device
        DeviceStrings['S21']=device + ' - S21'
        DeviceStrings['S21 subvalues']=dict()
        DeviceStrings['S21 subvalues']['y']='y'
        DeviceStrings['S21 subvalues']['t0']='t0'
        #if device=='STLSS_VNA_New':
        DeviceStrings['Center frequency']=device + ' - Center frequency'
        DeviceStrings['Span']=device + ' - Span'
        #else:
        DeviceStrings['Start frequency']=device + ' - Start frequency'
        DeviceStrings['Stop frequency']=device + ' - Stop frequency'
        DeviceStrings['Power']=device + ' - Output power'
        DeviceStrings['Average']=device + ' - Average'
        DeviceStrings['Number of averages']=device + ' - # of averages'
        DeviceStrings['IFBW']=device + ' - IF bandwidth'
    elif device=='hetrodyne':
        DeviceStrings=dict()
        DeviceStrings['device']=device
        DeviceStrings['S21']='Signal Demodulation with segments - Value'
        DeviceStrings['Power']='Readout Generator - Power'
        DeviceStrings['Freq']='Readout Generator - Frequency'
    else:
        raise ValueError('Bad device name')
    return DeviceStrings, autofind

def load_data(filename,resonator=None,delay=0,DeviceStrings='N5230C'):
    """
    Function that parses a Labber log file and returns the data in a list of
    datasets.
    Inputs:
        filename - Labber logfile filename
        resonator (None) - a previous list that results can be appended to
        delay (0) - Electrical delay
        DeviceStrings ('N5230C') - This variable can be a string or a dict. As 
                a dictionary this variable stores Labber driver channel names
                needed to retrieve information from the log file.
                If DeviceStrings is a string it will call the function
                labber_driver_default_names to load the driver's dictionary.
    Output:
        resonator - A list of datasets where each element contains the
                information from a trace.
    dataset class variable additions:
        dataset.time - Beginning time of trace
        dataset.filname - the filename used to parse this trace
        dataset.IFBW - if exists in log file this will add the IF bandwidth
        dataset.power - if exists in log file this will add the output power
    """
    
    # Include this line because python remembers function default mutable
    # variables.
    if resonator is None:
        resonator=[]
        
    log=Labber.LogFile(filename)
    Entry=log.getEntry(0)
    if type(DeviceStrings) == str:
        DeviceStrings,autofind=labber_driver_default_names(device=DeviceStrings,Entry=Entry)
    for EntryNumber in range(log.getNumberOfEntries()):
        Entry=log.getEntry(EntryNumber)
        if DeviceStrings['device']=='N5230C' or autofind or DeviceStrings['device']=='STLSS_VNA_New':        
            data=Entry[DeviceStrings['S21']]['y']
            averages=1
            if ((DeviceStrings['Average'] in Entry) and (DeviceStrings['Number of averages'] in Entry)):
                if Entry[DeviceStrings['Average']]:
                    averages=Entry[DeviceStrings['Number of averages']]
            if DeviceStrings['Span'] in Entry:
                span=Entry[DeviceStrings['Span']]
                centerfreq=Entry[DeviceStrings['Center frequency']]
                resonator.append(dataset(data,centerfreq=centerfreq,
                                 span=span,delay=delay,averages=averages))
            else:
                startfreq=Entry[DeviceStrings['Start frequency']]
                stopfreq=Entry[DeviceStrings['Stop frequency']]
                resonator.append(dataset(data,startfreq=startfreq,
                                 stopfreq=stopfreq,delay=delay,averages=averages))
            resonator[-1].time=Entry['timestamp']
            if DeviceStrings['IFBW'] in Entry:
                resonator[-1].IFBW=Entry[DeviceStrings['IFBW']]
            if DeviceStrings['Power'] in Entry:
                resonator[-1].power=Entry[DeviceStrings['Power']]
            resonator[-1].filename=filename
            # Stick the rest of the values into the 'Entry' value
            resonator[-1].Entry=copy.deepcopy(Entry)
        else:
            data=Entry[DeviceStrings['S21']]
            freq=Entry[DeviceStrings['Freq']]
            resonator.append(dataset(data,freq=freq,delay=delay))
            if DeviceStrings['Power'] in Entry:
                resonator[-1].power=Entry[DeviceStrings['Power']][0]
            resonator[-1].filename=filename
            resonator[-1].Entry=Entry   
        if not EntryNumber%10:
            print(EntryNumber)
    return resonator

def losstan(res,plot=True,yscale='log'):
    """
    Function that returns parameters from a list of datasets and plots the
    quality factor
    Inputs:
        res - A list of datasets
    Outputs:
        n - Number of photons in the resonator
        losstan - the loss tangent
        powers - The power at the resonator transmission line
        err_qi - the errors in the internal quality factor
        freqs - the resonator frequencies
        err_freqs - the error in the frequency fits
        time - the start time of the trace
        Qc - the coupling quality factor        
    """
    losstan=np.array([])
    n=np.array([])
    err_qi=np.array([])
    powers=np.array([])
    freqs=np.array([])
    err_freqs=np.array([])
    time=np.array([])
    qc=np.array([])
    hbar=1.05e-34
    for i in range(len(res)):
        if len(res[i].fit_x)==1:
            continue
        losstan=np.concatenate((losstan,[1/res[i].Qi]))
        power=10**((res[i].power-80)/10)*1e-3
        powers=np.concatenate((powers,[res[i].power]))
        photons=2*power*res[i].Q**2/(hbar*(2*np.pi*res[i].f0)**2*res[i].Qc)
        n=np.concatenate((n,[photons]))
        err_qi=np.concatenate((err_qi,[np.sqrt(res[i].pcov[1,1])]))
        err_freqs=np.concatenate((err_freqs,[np.sqrt(res[i].pcov[0,0])]))
        freqs=np.concatenate((freqs,[res[i].f0]))
        time=np.concatenate((time,[res[i].time]))
        qc=np.concatenate((qc,[res[i].Qc]))
    if plot:
        plt.figure()
        plt.clf()
        plt.errorbar(n,1/losstan,yerr=err_qi,fmt='.')
        plt.xscale('log')
        plt.yscale(yscale)
        plt.ylabel('Qi')
        plt.xlabel('n')
        plt.ylim(0.9/losstan.max(),1.1/losstan.min())
        plt.title(res[-1].f0)
        plt.show()
    return(n,losstan,powers,err_qi,freqs,err_freqs,time,qc)

def losstan_fit(n,losstan,weight=1, bnds=[[2e-7,2e-3],[1e-5,1e5],[.1,2],[0,1e-3]],\
                x0=[1e-4, 1, 1, 0],cov_return=False):
    """
    Function that fits to the loss tangent power dependance. The form is:
    x[0]/np.sqrt(1+n/x[1])**x[2]+x[3]
    
    Inputs:
        n - photon number
        losstan - loss tangent curve
        weight - (optional) standard deviation of loss tangents
        bnds - (optional) Bounds for the fiting parameters
        x0 - (optional) Initial values
    Outputs:
        n - Number of photons in the resonator
        losstan - the loss tangent
        powers - The power at the resonator transmission line
        err_qi - the errors in the internal quality factor
        freqs - the resonator frequencies
        err_freqs - the error in the frequency fits
        time - the start time of the trace
        Qc - the coupling quality factor        
    """
    
    
    def lstsqr(x,n,losstan,weight,fittype='log'):
        weights=weight*1
        if fittype=='log':
            fit=np.log(x[0]/(1+n/10**x[1])**(x[2]/2)+x[3])
            data=np.log(losstan)

            if type(weights) is not int:
                weights=np.divide(weights,losstan)
        else:
            fit=x[0]/(1+n/10**x[1])**(x[2]/2)+x[3]
            data=losstan
        
        lstsqr=np.sum((fit-data)**2/weights**2)
        return lstsqr
    
    
    fctr=1e6
        
    bounds=copy.deepcopy(bnds)
    bounds[0][0]*=fctr
    bounds[0][1]*=fctr
    bounds[1][0]=np.log10(bounds[1][0])
    bounds[1][1]=np.log10(bounds[1][1])
    bounds[3][0]*=fctr
    bounds[3][1]*=fctr
    x0s=copy.copy(x0)
    x0s[0]*=fctr
    x0s[1]=np.log10(x0s[1])
    x0s[3]*=fctr
    losstans=fctr*np.array(losstan)
    x=scipy.optimize.minimize(lstsqr,x0=x0s,#jac=jacobian,hess=hessian,\
                              bounds=bounds, args=(n,losstans,weight),options={'disp': 2})
#    x=scipy.optimize.minimize(lstsqr,x0=x0,#jac=jacobian,hess=hessian,\
#                              bounds=bnds, args=(n,losstan,weight),options={'disp': 2})
    print(x)
    fit=np.array(x['x'])
    cov=np.matrix(x['hess_inv'].todense())
    J=np.matrix([[1/fctr,0,0,0],[0,1,0,0],\
                  [0,0,1,0],[0,0,0,1/fctr]])
    fit[0]/=fctr
    fit[1]=fit[1]
    fit[3]/=fctr
#    cov=np.linalg.inv(hessian(fit,n,losstan,weight,fittype='lin'))
    cov=J@cov@J
    print('----------------------------')
    print(fit)
    print([np.sqrt(cov[k,k]) for k in range(4)])
    print('----------------------------')
    plt.clf()
    plt.loglog(n,losstan,'.',n,losstan_fit_curve(n,fit),'-r')
    plt.title('Tand0 = %.2e' %fit[0] + \
        ', n_0 = %.2e' %fit[1]+\
        ', tand_{anom} = %.2e, beta = %f'%(fit[3],fit[2]))
    plt.xlabel('n')
    plt.ylabel('tan delta')
    if cov_return:
        return fit, cov
    else:
        return fit
    
def losstan_fitv2(n,losstan,weight=None, bnds=(np.array([2e-7,1e-5,.1,0]),\
                                            np.array([2e-3,1e5,2,1e-3])),\
                x0=[1e-4, 1, 1, 0],cov_return=False):
    """
    Function that fits to the loss tangent power dependance. The form is:
    x[0]/np.sqrt(1+n/x[1])**x[2]+x[3]
    
    Inputs:
        n - photon number
        losstan - loss tangent curve
        weight - (optional) standard deviation of loss tangents
        bnds - (optional) Bounds for the fiting parameters
        x0 - (optional) Initial values
    Outputs:
        n - Number of photons in the resonator
        losstan - the loss tangent
        powers - The power at the resonator transmission line
        err_qi - the errors in the internal quality factor
        freqs - the resonator frequencies
        err_freqs - the error in the frequency fits
        time - the start time of the trace
        Qc - the coupling quality factor        
    """
    
    
    def lstsqr(n,d0,n0,b,dh):
        fit=np.log(d0/(1+n/n0)**(b/2)+dh)
        return fit
    
    if weight is not None:
        weight/=losstan
    x=scipy.optimize.curve_fit(lstsqr,n,np.log(losstan),p0=x0,#jac=jacobian,hess=hessian,\
                              bounds=bnds,sigma=weight)
    print('-----------------------')
    print(x)
    fit=x[0]
    cov=x[1]
    plt.clf()
    plt.loglog(n,losstan,'.',n,losstan_fit_curve(n,fit),'-r')
    plt.title('Tand0 = %.2e' %fit[0] + \
        ', n_0 = %.2e' %fit[1]+\
        ', tand_{anom} = %.2e, beta = %f'%(fit[3],fit[2]))
    plt.xlabel('n')
    plt.ylabel('tan delta')
    if cov_return:
        return fit, cov
    else:
        return fit
def losstan_fit_moshe(n,losstan,weight=None, bnds=(np.array([2e-7,1e-5,0,0,1e-5]),\
                                            np.array([2e-3,1e5,1e-3,2e-3,1e5])),\
                x0=[1e-4, 1, 1, 0],cov_return=False):
    """
    Function that fits to the loss tangent power dependance. The form is:
    x[0]/np.sqrt(1+n/x[1])**x[2]+x[3]
    
    Inputs:
        n - photon number
        losstan - loss tangent curve
        weight - (optional) standard deviation of loss tangents
        bnds - (optional) Bounds for the fiting parameters
        x0 - (optional) Initial values
    Outputs:
        n - Number of photons in the resonator
        losstan - the loss tangent
        powers - The power at the resonator transmission line
        err_qi - the errors in the internal quality factor
        freqs - the resonator frequencies
        err_freqs - the error in the frequency fits
        time - the start time of the trace
        Qc - the coupling quality factor        
    """
    
    
    def lstsqr(n,d0,n0,dh,d1,n1):
        fit=np.log(d0/(1+n/n0)**(1/2)+dh+d1/(1+n/n1)**(1/2))
        return fit
    
    if weight is not None:
        weight/=losstan
    x=scipy.optimize.curve_fit(lstsqr,n,np.log(losstan),p0=x0,#jac=jacobian,hess=hessian,\
                              bounds=bnds,sigma=weight)
    print('-----------------------')
    print(x)
    fit=x[0]
    cov=x[1]
    plt.clf()
    plt.loglog(n,losstan,'.',n,losstan_fit_curve(n,fit),'-r')
    plt.title('Tand0 = %.2e' %fit[0] + \
        ', n_0 = %.2e' %fit[1]+\
        ', tand_{anom} = %.2e, beta = %f'%(fit[3],fit[2]))
    plt.xlabel('n')
    plt.ylabel('tan delta')
    if cov_return:
        return fit, cov
    else:
        return fit

def losstan_fit_curve(n,fit):
    """
    Function return the power dependence of losstangents. The form is:
    x[0]/np.sqrt(1+n/x[1])**x[2]+x[3]
    
    Inputs:
        n - photon number
        fit - the fit parameters
    Outputs:
        losstan - loss tangent
    """
    loss=fit[0]/np.sqrt(1+n/fit[1])**fit[2]+fit[3]
    return loss

def losstan_fit_curve_moshe(n,fit):
    """
    Function return the power dependence of losstangents. The form is:
    x[0]/np.sqrt(1+n/x[1])**x[2]+x[3]
    
    Inputs:
        n - photon number
        fit - the fit parameters
    Outputs:
        losstan - loss tangent
    """
    loss=fit[0]/np.sqrt(1+n/fit[1])+fit[2]+fit[3]/np.sqrt(1+n/fit[4])
    return loss
    
    
def test_labber_log_resonators():
    """
    Test of the general functionality of the above code for users to see how the
    code might work for their data
    """    

    # Start with the log file name
    filename = 'myfilename.hdf5'

    # Get a 'dataset' object from the filename   
    dset = load_data(filename)

    # Plot S21 data in dB    
    dset[0].plot()

    # Save the data
    save(dset,'mydset.bin')


if __name__ == '__main__':
    test_labber_log_resonators()
