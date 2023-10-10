# -*- coding: utf-8 -*-
"""
Title: S-Curve Projection Functions
Created on Sun Aug 13 05:42:19 2023
Description: Library with S-curve projections in function form

@author: Donovin Lewis
"""

import numpy as np #conda install numpy
#conda install matplotlib

def scurve_fit(data_array, plot_flag = False):
    '''
    Attempts to fit data to s-curve. Returns coefficients. Set plot_flag to 
    True to view fit on log-log plot

    Parameters
    ----------
    data_array : Numpy Array
        Array with timesteps in the first row and data in the second.
    plot_flag : Boolean, optional
        Set True to view fit on log-log plot. The default is False.

    Returns
    -------
    coeffs : Array of floats
        Polynomial coefficients for s-curve projection.

    '''
    
    
    y = (1/data_array[1,:]) - 1 #Solve for e^(-(ax +b))
    y_log = -1 * np.log(y) #Apply log to cancel out e
    
    x_log = np.log(data_array[0,:]) #Convert time steps to log for fit
    
    coeffs = np.polyfit(x_log, y_log, 1) #Fit polynomial to log-log trend
    
    if plot_flag == True:  #Plot log-log curve
        import matplotlib.pyplot as plt #conda install matplotlib
        import matplotlib as mpl

        mpl.style.use('classic')
        FIGUREWIDTH = 3.3 #inches; this is used to control the figure width
        PROPORTION = 0.62
        LABELFONTSIZE = 7
        LINEWIDTH = 0.1
        TICKSIZE = 2
        mpl.rcParams['xtick.major.size'] = TICKSIZE
        mpl.rcParams['ytick.major.size'] = TICKSIZE
        mpl.rcParams['axes.labelsize'] = LABELFONTSIZE
        mpl.rcParams['xtick.labelsize'] = LABELFONTSIZE
        mpl.rcParams['ytick.labelsize'] = LABELFONTSIZE
        plt.rcParams["figure.figsize"] = (FIGUREWIDTH,FIGUREWIDTH*PROPORTION)
        
        
        fig, ax = plt.subplots(1,1, sharex=True, sharey = True, figsize=(FIGUREWIDTH,FIGUREWIDTH*PROPORTION), constrained_layout=False)
        plt.scatter(x_log, y_log)
        
        p = np.poly1d(coeffs)
        plt.plot(x_log, p(x_log), '--')
        ax.ticklabel_format(useOffset=False)
        plt.legend(['Original', 'Linear Fit'], fontsize = LABELFONTSIZE, frameon = False, loc = 'upper left')
        plt.text(0.6, 0.1 , f'y = {coeffs[0]:.0f}x + {coeffs[1]:.0f}', fontsize=LABELFONTSIZE, transform = ax.transAxes)
        
        
        fig, ax = plt.subplots(1,1, sharex=True, sharey = True, figsize=(FIGUREWIDTH,FIGUREWIDTH*PROPORTION), constrained_layout=False)
        plt.plot(data_array[0,:], data_array[1,:] * 100, label = 'True')
        
        projected_array = (1/(1 + np.exp(-coeffs[0] * x_log - coeffs[1])))
        plt.plot(np.exp(x_log), projected_array * 100, linestyle = '--', label = 'Predicted')
        
        ax.ticklabel_format(useOffset=False)
        plt.xlim([min(data_array[0,:]), max(data_array[0,:])])
        plt.ylabel('Technology Adoption [%]', size = LABELFONTSIZE)
        plt.xlabel('Time [yrs]', size = LABELFONTSIZE)
        plt.legend(['Original', 'Polynomial Fit'], fontsize = LABELFONTSIZE, frameon = False, loc = 'upper left')
    
    return coeffs
    

    
def scurve_project(data_array, coeffs, SAMPLE_NUMBER = 25, plot_flag = False):
    '''
    Project s-curve adoption with fit coefficients a set number of years

    Parameters
    ----------
    data_array : Numpy Array
        Timesteps as the first row, data points as the second.
    coeffs : Numpy array
        Coefficients from polynomial fit in log-log form.
    SAMPLE_NUMBER : Integer, optional
        Number of time steps to project into the future. The default is 25.
    plot_flag : Boolean, optional
        Set True to view fit on log-log plot. The default is False.

    Returns
    -------
    Numpy Array
        Projected data with time steps as the first row, data points as the second.

    '''
    

    sample_array = np.arange(data_array[0,:][-1], data_array[0,:][-1] + SAMPLE_NUMBER)
    sample_log = np.log(sample_array)
    
    projected_array = (1/(1 + np.exp(-coeffs[0] * sample_log - coeffs[1])))
    
    if plot_flag == True:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        #Plot Style Modification in Matplotlib
        mpl.style.use('classic')
        # define plots settings
        FIGUREWIDTH = 3.3 #inches; this is used to control the figure width
        PROPORTION = 0.62
        LABELFONTSIZE = 7
        LINEWIDTH = 0.1
        TICKSIZE = 2
        mpl.rcParams['xtick.major.size'] = TICKSIZE
        mpl.rcParams['ytick.major.size'] = TICKSIZE
        mpl.rcParams['axes.labelsize'] = LABELFONTSIZE
        mpl.rcParams['xtick.labelsize'] = LABELFONTSIZE
        mpl.rcParams['ytick.labelsize'] = LABELFONTSIZE
        plt.rcParams["figure.figsize"] = (FIGUREWIDTH,FIGUREWIDTH*PROPORTION)
        
        
        fig, ax = plt.subplots(1,1, sharex=True, sharey = True, figsize=(FIGUREWIDTH,FIGUREWIDTH*PROPORTION), constrained_layout=False)
        plt.plot(sample_array, projected_array, linestyle = '--')
        ax.ticklabel_format(useOffset=False)
        plt.xlim([min(sample_array), max(sample_array)])
        plt.ylabel('Technology Adoption [%]', size = LABELFONTSIZE)
        plt.xlabel('Time [yrs]', size = LABELFONTSIZE)
        plt.legend(['Original', 'Projection'], loc = 'upper left', fontsize = LABELFONTSIZE, frameon = False)
        
        
    return np.array([sample_array, projected_array])



def scurve_plot(data_array, legend_array = None):
    '''
    S-curve plotting function

    Parameters
    ----------
    data_array : Numpy Array
        Time steps as the first row, data points as the following rows.
        Multiple rows of data will be plotted on the same x-axis
    legend_array : List of strings, optional
        List of string descriptors for the legend. The default is None.

    Returns
    -------
    None.

    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #Plot Style Modification in Matplotlib
    mpl.style.use('classic')
    # define plots settings
    FIGUREWIDTH = 3.3 #inches; this is used to control the figure width
    PROPORTION = 0.62
    LABELFONTSIZE = 7
    LINEWIDTH = 0.1
    TICKSIZE = 2
    mpl.rcParams['xtick.major.size'] = TICKSIZE
    mpl.rcParams['ytick.major.size'] = TICKSIZE
    mpl.rcParams['axes.labelsize'] = LABELFONTSIZE
    mpl.rcParams['xtick.labelsize'] = LABELFONTSIZE
    mpl.rcParams['ytick.labelsize'] = LABELFONTSIZE
    plt.rcParams["figure.figsize"] = (FIGUREWIDTH,FIGUREWIDTH*PROPORTION)
    
    fig, ax = plt.subplots(1,1, sharex=True, sharey = True, figsize=(FIGUREWIDTH,FIGUREWIDTH*PROPORTION), constrained_layout=False)
    
    for i in np.arange(1, np.size(data_array, 0)):
        plt.plot(data_array[0,:], data_array[i,:] * 100)
        if legend_array != None:
            plt.legend(legend_array, fontsize = LABELFONTSIZE, frameon = False, loc = 'upper left')
    
    ax.ticklabel_format(useOffset=False)
    plt.xlim([min(data_array[0,:]), max(data_array[0,:])])
    plt.ylim([0, 100])
    plt.ylabel('Technology Adoption [%]', size = LABELFONTSIZE)
    plt.xlabel('Time [yrs]', size = LABELFONTSIZE)
    

if __name__ == '__main__':

    #Load data and create the numpy array
    years = [ 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    pcts = [0.0001, 0.0002, 0.0003, 0.0005, 0.0006, 0.001, 0.0019]    
    kentucky_data = np.array([years, pcts])
    plot_flag = True
    SAMPLE_NUMBER = 25
    

    coeffs = scurve_fit(kentucky_data, plot_flag) #Create the polynomial fit using the log-log projection
    kentucky_results = scurve_project(kentucky_data, coeffs, SAMPLE_NUMBER) #Project for SAMPLE_NUMBER time steps using the coefficients
    
    coeffs_four = scurve_fit(kentucky_data[:, -4:])
    kentucky_four = scurve_project(kentucky_data[:, -4:], coeffs_four, SAMPLE_NUMBER)
    
    kentucky_total = np.vstack((kentucky_results, kentucky_four[1,:])) #Vertically stack the numpy array to put multiple results on the same axis
    
    scurve_plot(kentucky_total, ['Full', 'Last Four']) #Plot both curves on the same axis

    