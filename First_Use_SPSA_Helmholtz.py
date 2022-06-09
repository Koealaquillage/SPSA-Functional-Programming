#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:30:52 2018

@author: koenig.g
"""

#############################################
# First try of implementation of an SPSA met#
# -hod to compute initial conditions of an  #
# Differential equation. We're going to see #
# What we get. To compare we are going to us#
# e a total inversion scheme.               #
# Try with a real Helmholtz equation        #
#############################################

#*******Adding path before******************#
#*******The package import******************#
import sys
sys.path.append('../SERPENT_DE_MER/LIBRARIES/') #For Serpent_De_Mer
sys.path.append('LIBRARIES/') # For the SPSA module
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
            'a':6e-3,'c':0.9,'gamma':0.4,'J':J_SPSA}
#*************Initializing values******************#
    
Test=seasnake.SerpentdeMerMatricial() # Create a Serpent de  Mer instance

Test.set_grid_manually(20,20,np.linspace(20,25,20),np.linspace(20,25,20),
                       dx=1.,dy=1.,h=1.)

#********Create grid and boundary conditions*****#


Test.create_solving_matrix_overelevation(g=1.,sigma=1.) 
# The resolution matrix, now  called Test.A
Test.create_projection_matrix_overelevation(g=1.,sigma=1.) 
# The matrix that sends initial conditions, now Called Test.B
Test.create_bc_vector_overelevation() 
# The boundary condition vector, now called Test.Y
Test.create_solution_vector_overelevation() 
# The solution vector, now called Test.X

#*************Now we can solve********************#

Test.X_eta=scipy.sparse.linalg.lsqr(Test.A_eta,Test.B_eta*Test.Y_eta)
Test.X_eta=Test.X_eta[0]
#***********Now that we have solved it************#
#***********We can compute the inverse************#

Time_inv = timeit.default_timer() # To measure execution time

Y_inv=scipy.sparse.linalg.lsqr(Test.B_eta,Test.A_eta*Test.X_eta) 
# We're gonna used a least square estimating solver for the 
# System of equations

Time_inv=timeit.default_timer()-Time_inv 

#**********And try the SPSA method on it**********#

Y_SPSA=np.array([random.random() for p in range(Test.Y_eta.size)])
Y_SPSA_One=Y_SPSA.copy()
Y_SPSA_Mean=Y_SPSA.copy()
print(Y_SPSA)

#Cost_func=lambda y :np.dot(Test.A_eta*Test.X_eta-Test.B_eta*y,
#            (Test.A_eta*Test.X_eta-Test.B_eta*y))

Cost_func=lambda y : np.dot(Test.Y_eta-y,Test.Y_eta-y)
SPSA_param.update({'func':Cost_func,'theta':Y_SPSA})
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
SPSA_param.update({'theta':Y_SPSA_Mean,'J':J_SPSA_Mean,'n_mean':4}) 
# List of parameters  

Time_SPSA_Mean=timeit.default_timer()             
Y_SPSA_Mean,J_SPSA_Mean=SPSA.Compute_SPSA_Mean(**SPSA_param)
Time_SPSA_Mean=timeit.default_timer()-Time_SPSA_Mean

Y_inv_cost=Cost_func(Y_inv[0])

#********Visualizing part*********************#
fig_y_real=plt.figure()
ax_y_real=fig_y_real.add_subplot(1,1,1)
ax_y_real.plot(Test.Y_eta,marker='*',
               linestyle='-',label='Initial value')
ax_y_real.plot(Y_SPSA,marker='o',
               linestyle='--',label='SPSA method (Standard)'+
               str(Time_SPSA)+' seconds')
ax_y_real.plot(Y_SPSA_One,marker='s',
               linestyle=':',label='SPSA method (One_sided)'+
               str(Time_SPSA_One)+' seconds')
ax_y_real.plot(Y_SPSA_Mean,marker='p',
               linestyle='-.',label='SPSA method (Mean) '+
               str(Time_SPSA_Mean)+' seconds')
ax_y_real.plot(Y_inv[0],marker='*',
               linestyle=':',label='Inverse method '+
               str(Time_inv)+' seconds')
ax_y_real.legend(loc='best')
ax_y_real.set_xlabel('# of vector component')
ax_y_real.set_ylabel('Y value(real)')


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
fig_J.show()