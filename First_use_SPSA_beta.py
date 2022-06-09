#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:33:52 2018

@author: koenig.g
"""

#############################################
# First try of implementation of an SPSA met#
# -hod to compute initial conditions of an  #
# Differential equation. We're going to see #
# What we get. To compare we are going to us#
# e a total inversion scheme.               #
# In a first time we will work only in the  #
# Real part, since it is gonna be tricky to #
# Implement the imaginary part              #
# Added parallel formulation of SPSA method #
# 30/05/2018                                #
#############################################

#*******The package import******************#
import sys
sys.path.append('../SERPENT_DE_MER/LIBRARIES/') # For Serpent_de_MER
sys.path.append('LIBRARIES/') # For SPSA methods

#********Packages Import********************#

import numpy as np
import matplotlib.pyplot as plt
import random,timeit,scipy
import Serpent_de_Mer_Matricial as seasnake 
# My package for hydrodynamic modelling
import libspsa as SPSA # My package for SPSA module

#********Variables declarations*************#
Test=[] # Serpent de Mer instance, for computations

Y_inv=[]
Y_inv_cost=[] # To evaluate the cost function of the inverse vector
Y_SPSA=[] # To store resolution
Y_SPSA_One=[]
Y_SPSA_Mean=[] # To store alternative SPSA models

Time_inv=0.
Time_SPSA=0. # To store the time for using the SPSA and total inverse method
Time_SPSA_One=[]
Time_SPSA_Mean=[]


J_SPSA=[]
J_SPSA_One=[] 
J_SPSA_Mean=[]
# To store the cost function of SPSA iterations
# Parameters for the SPSA solver
SPSA_param={'n_iter':10000,'eps':1e-3,'alpha':0.3,'A':1e-4,
            'a':6e-3,'c':0.9,'gamma':0.4,'J':J_SPSA,'imag':True}
#*************Initializing values******************#
    
Test=seasnake.SerpentdeMerMatricial() # Create a Serpent de  Mer instance

Test.set_grid_manually(20,20,np.linspace(20,25,20),np.linspace(20,25,20),
                       dx=2.,dy=1.,h=1.)


#********Create grid and boundary conditions*****#

Test.create_solving_matrix(g=1.,K=1.,sigma=1.) 
# The resolution matrix, now  called Test.A
Test.create_projection_matrix(g=1.,K=1.,sigma=1.) 
# The matrix that sends initial conditions, now
# Called Test.B
Test.create_bc_vector() # The boundary condition vector, now called Test.Y
Test.create_solution_vector_velocity() # The solution vector, now called Test.X

#*************Now we can solve********************#
Test.X=scipy.sparse.linalg.spsolve(Test.A,Test.B*Test.Y)

#***********Now that we have solved it************#
#***********We can compute the inverse************#

Time_inv = timeit.default_timer() # To measure execution time
Y_inv=scipy.sparse.linalg.lsqr(Test.B,Test.A*Test.X)
Y_inv=Y_inv[0] # So that I do not save the additional values it gives me

Time_inv=timeit.default_timer()-Time_inv 

#**********And try the SPSA method on it**********#


Y_SPSA=Test.Y.copy()+np.array([random.random()+random.random()*1j 
                   for p in range(Test.Y.size)])*0.5
Y_SPSA_One=Y_SPSA.copy()
Y_SPSA_Mean=Y_SPSA.copy()
print(Y_SPSA)

Cost_func=lambda y :np.dot(Test.Y-y,(Test.Y-y).conj())

SPSA_param.update({'theta':Y_SPSA,'func': Cost_func}) # List of parameters  
            #for the SPSA method, to adjust
#### Standard SPSA############

Time_SPSA=timeit.default_timer()             
Y_SPSA,J_SPSA=SPSA.Compute_SPSA(**SPSA_param)# the ** is used to
# Call a dictionnary in the argument
Time_SPSA=timeit.default_timer()-Time_SPSA 

#### One_sided SPSA############
SPSA_param.update({'theta':Y_SPSA_One,'J':J_SPSA_One}) # List of parameters  
            #for the SPSA method, to adjust

Time_SPSA_One=timeit.default_timer()             
Y_SPSA_One,J_SPSA_One=SPSA.Compute_SPSA_one_sided(**SPSA_param)
Time_SPSA_One=timeit.default_timer()-Time_SPSA_One

#### Mean SPSA############
SPSA_param.update({'n_mean':4,'theta':Y_SPSA_Mean,'J':J_SPSA_Mean}) 
# List of parameters  
            #for the SPSA method, to adjust

Time_SPSA_Mean=timeit.default_timer()             
Y_SPSA_Mean,J_SPSA_Mean=SPSA.Compute_SPSA_Mean(**SPSA_param)
Time_SPSA_Mean=timeit.default_timer()-Time_SPSA_Mean

Y_inv_cost=Cost_func(Y_inv)

#********Visualizing part*********************#
fig_y_real=plt.figure()
ax_y_real=fig_y_real.add_subplot(1,1,1)
ax_y_real.plot(Test.Y.real,marker='*',
               linestyle='-',label='Initial value')
ax_y_real.plot(Y_SPSA.real,marker='o',
               linestyle='--',label='SPSA method (Standard)'+str(Time_SPSA)
               +' seconds')
ax_y_real.plot(Y_SPSA_One.real,marker='s',
               linestyle=':',label='SPSA method (One_sided)'+str(Time_SPSA_One)
               +' seconds')
ax_y_real.plot(Y_SPSA_Mean.real,marker='p',
               linestyle='-.',label='SPSA method (Mean) '+str(Time_SPSA_Mean)
               +' seconds')
ax_y_real.plot(Y_inv.real,marker='*',
               linestyle=':',label='Inverse method '+str(Time_inv)+' seconds')
ax_y_real.legend(loc='best')
ax_y_real.set_xlabel('# of vector component')
ax_y_real.set_ylabel('Y value(real)')

fig_y_imag=plt.figure()
ax_y_imag=fig_y_imag.add_subplot(1,1,1)
ax_y_imag.plot(Test.Y.imag,marker='*',
               linestyle='-',label='Initial value')
ax_y_imag.plot(Y_SPSA.imag,marker='o',
               linestyle='--',label='SPSA method (Standard)'+str(Time_SPSA)
               +' seconds')
ax_y_imag.plot(Y_SPSA_One.imag,marker='s',
               linestyle=':',label='SPSA method (One_sided)'+str(Time_SPSA_One)
               +' seconds')
ax_y_imag.plot(Y_SPSA_Mean.imag,marker='p',
               linestyle='-.',label='SPSA method (Mean) '+str(Time_SPSA_Mean)
               +' seconds')
ax_y_imag.plot(Y_inv.imag,marker='*',
               linestyle=':',label='Inverse method '+str(Time_inv)+' seconds')
ax_y_imag.legend(loc='best')
ax_y_imag.set_xlabel('# of vector component')
ax_y_imag.set_ylabel('Y value(imaginary)')

fig_J=plt.figure()
ax_J=fig_J.add_subplot(1,1,1)
ax_J.plot(J_SPSA,marker='o',label='SPSA ( Standard)')
ax_J.plot(J_SPSA_One,marker='p',label='SPSA ( One_sided)')
ax_J.plot(J_SPSA_Mean,marker='s',label='SPSA ( Mean)')
ax_J.plot(np.ones(J_SPSA.size)*Y_inv_cost,label='Inverse method')
ax_J.legend(loc='best')
ax_J.set_xlabel('Iteration')
ax_J.set_ylabel('Cost function')

fig_y_real.show()
fig_y_imag.show()
fig_J.show()