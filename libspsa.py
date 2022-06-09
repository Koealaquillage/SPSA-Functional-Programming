#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:16:51 2018

@author: koenig.g
"""

###############################
# Libraries to store all SPSA #
# Functions. First not entirel#
# -y optimize.                #
###############################

#****Packages import**********#

import numpy as np
import matplotlib.pyplot as plt
import random

#****SPSA functions***********#

def Compute_SPSA(n_iter,eps,alpha,A,a,c,gamma,theta,func,J,imag=False) :
    """ A function that computes n_iter of the cost 
        function using the SPSA method. Note that the stop criterion
        is only effective if the optimal cost function is 0. This will
        not always be the case. A more effective case would be to test the
        hessian or the gradient.
        INPUTS:
        n_iter : Integer. Number of iterations of the method
        eps : Precision needed for the cost function
        alpha : Float. Power of the gain function
        A : Float. Initial gain coefficient.
        a : Float. Initial gain value.
        c : Float. Distance between iteration.
        gamma : Float. Power to get the distance between iterations
        theta : Vector of floats. Data input.
        func : A function used to determine the cost function
        J: Cost function estimation vector. To see how it evolves
        imag: Flag for imaginary data input
        OUTPUTS :
        theta : Specified above"""
    
    J=np.zeros(n_iter)
    # Cost function of initial conditions
    J[0]=func(theta)
    
    # Iterations
    for k in range(1,n_iter):
        # Setting the gain  values
        ak=a/((k+A)**alpha)
        ck=c/(k**gamma)
        # Perturbed vector creation    
        delta=(np.array([random.randint(0,1) for p in \
                         range(theta.size)],dtype=complex)-0.5)*2. # Real part
        if imag:
           delta+=(np.array([random.randint(0,1) for p in \
                         range(theta.size)],dtype=complex)-0.5)*2.*1j
                         # Imaginary part
        thetaplus=theta+ck*delta
        thetaminus=theta-ck*delta
        # Now we evaluate the cost function
        yplus=func(thetaplus)
        yminus=func(thetaminus)
        # Gradient estimating 
        ghat=(yplus-yminus)/(2*ck*delta)
        theta=theta-ak*ghat

        # Storing the cost function
        J[k]=func(theta)
        #Check convergence of the SPSA method
        # We omit the possible little rest of imaginary part
        if J[k]<eps:
            break # We break the circle then
    
    return theta,J
def Compute_SPSA_one_sided(n_iter,eps,alpha,A,a,c,gamma,theta,func,J
                           ,imag=False) :
    """ A function that computes n_iter of the cost 
        function using the SPSA method. Note that the stop criterion
        is only effective if the optimal cost function is 0. This will
        not always be the case. A more effective case would be to test the
        hessian or the gradient. Here the SPSA is computed with a one-sided 
        formulatin of the gradient given in Spall 1998.
        INPUTS:
        n_iter : Integer. Number of iterations of the method
        eps : Precision needed for the cost function
        alpha : Float. Power of the gain function
        A : Float. Initial gain coefficient.
        a : Float. Initial gain value.
        c : Float. Distance between iteration.
        gamma : Float. Power to get the distance between iterations
        theta : Vector of floats. Data input.
        func : A function used to determine the cost function
        J: Cost function estimation vector. To see how it evolves
        imag: Flag for imaginary data input
        OUTPUTS :
        theta : Specified above"""
    
    J=np.zeros(n_iter)   
    # Cost function of initial conditions
    J[0]=func(theta)
    
    # Iterations
    for k in range(1,n_iter):
        # Setting the gain  values
        ak=a/((k+A)**alpha)
        ck=c/(k**gamma)
        # Perturbed vector creation    
        delta=(np.array([random.randint(0,1) for p in \
                         range(theta.size)],dtype=complex)-0.5)*2. # Real part
        if imag:
            delta+=(np.array([random.randint(0,1) for p in \
                         range(theta.size)],dtype=complex)-0.5)*2.*1j  
                         # Imaginary part
        
        thetaplus=theta+ck*delta
        # Now we evaluate the cost function
        yplus=func(thetaplus)
        # Gradient estimating 
        ghat=(yplus-func(theta))/(ck*delta)
        theta=theta-ak*ghat
        
        # Storing the cost function
        J[k]=func(theta)
        #Check convergence of the SPSA method
        if J[k]<eps:
            break # We break the circle then
    
    return theta,J

def Compute_SPSA_Mean(n_iter,n_mean,eps,alpha,A,a,c,gamma,theta,func,J
                      ,imag=False) :
    """ A function that computes n_iter of the cost 
        function using the SPSA method. Note that the stop criterion
        is only effective if the optimal cost function is 0. This will
        not always be the case. A more effective case would be to test the
        hessian or the gradient. Here the gradient is taken as the mean of
        many iterations of the gradient ( Spall 1992).
        INPUTS:
        n_iter : Integer. Number of iterations of the method
        n_mean : Number of samples for the mean of the gradient
        eps : Precision needed for the cost function
        alpha : Float. Power of the gain function
        A : Float. Initial gain coefficient.
        a : Float. Initial gain value.
        c : Float. Distance between iteration.
        gamma : Float. Power to get the distance between iterations
        theta : Vector of floats. Data input.
        func : A function used to determine the cost function
        J: Cost function estimation vector. To see how it evolves
        imag: Flag for imaginary data input
        
        OUTPUTS :
        theta : Specified above"""
    
    J=np.zeros(n_iter)   
    # Cost function of initial conditions
    J[0]=func(theta)
    
    # Iterations
    for k in range(1,n_iter):
        #  Preparing the n iteratin for the mean
        for i in range(n_mean):
            # Setting the gain  values
            ak=a/((k+A)**alpha)
            ck=c/(k**gamma)
            # Perturbed vector creation    
            delta=(np.array([random.randint(0,1) for p in \
                         range(theta.size)],dtype=complex)-0.5)*2. # Real part
            if imag:
                delta+=(np.array([random.randint(0,1) for p in \
                         range(theta.size)],dtype=complex)-0.5)*2.*1j 
                         # Imaginary part
            thetaplus=theta+ck*delta
            thetaminus=theta-ck*delta
            # Now we evaluate the cost function
            yplus=func(thetaplus)
            yminus=func(thetaminus)
            # Gradient estimating 
            if i==0:
                ghat=(yplus-yminus)/(2*ck*delta)
            else :
                ghat+=(yplus-yminus)/(2*ck*delta)
    # Taking the mean operation, I do not know I have to use n_mean or n_mean-1
        ghat/=n_mean
        theta=theta-ak*ghat
        
        # Storing the cost function
        J[k]=func(theta)
        #Check convergence of the SPSA method
        # We omit the possible little rest of imaginary part
        if J[k]<eps:
            break # We break the circle then
    
    return theta,J