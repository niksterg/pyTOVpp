
# coding: utf-8

# 
# **TOV Stars with Piecewise Polytropic equation of state**
# 
# N. Stergioulas
# 
# Aristotle University of Thessaloniki
# 
# v1.0 (June 2018)
# 
# ###### Content provided under a Creative Commons Attribution license, 
# [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); 
# code under [GNU GPLv3 License](https://choosealicense.com/licenses/gpl-3.0/). 
# (c)2018 [Nikolaos Stergioulas](http://www.astro.auth.gr/~niksterg/)
# 

#
# Input: log10(p1[CGS])  Gamma1  Gamma2  Gamma3  rho_c[g/cm^3] 
#
# Output: rho_c/c^2[g/cm^3], eps_c[g/cm^3], Mass[Msun], M0[Msun], Radius[km], N_gridpoints
#

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import scipy as sp
from scipy import integrate
from scipy import optimize
from scipy.interpolate import PchipInterpolator
import sys
from decimal import Decimal
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
import contextlib

# these functions are used to suppress lsoda writing to warnings to stdout

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


# Constants and units

c=2.9979e10
G=6.67408e-8
Msun=1.989e33
Length = G*Msun/c**2
Time = Length/c
Density = Msun/Length**3


# command-line options

import argparse

parser = argparse.ArgumentParser(description='TOV solution for Piecewise Polytropic EOS')
parser.add_argument('p1', type=float, default=34.669,
                    help='p1')
parser.add_argument('Gamma1', type=float, default=2.909,
                    help='Gamma1')
parser.add_argument('Gamma2', type=float, default=2.246,
                    help='Gamma2')
parser.add_argument('Gamma3', type=float, default=2.144,
                    help='Gamma3')
parser.add_argument('rho_c', type=float, default=1e15,
                    help='central density')

args = parser.parse_args()

rho_c = args.rho_c/Density

# ## Define the equation of state

# Define the dividing densities for the high-density part

rho1 = pow(10,14.7)/Density
rho2 = pow(10,15.0)/Density


# Set $p_1, \Gamma_1, \Gamma_2, \Gamma_3$ for EOS

#SLy
#p1 = pow(10.0,34.384)/Density/c**2
#Gamma1 = 3.005
#Gamma2 = 2.988
#Gamma3 = 2.851

#H4
p1 = pow(10.0,args.p1)/Density/c**2
Gamma1 = args.Gamma1
Gamma2 = args.Gamma2
Gamma3 = args.Gamma3


# Find $K_1, K_2, K_3$

K1 = p1 / pow(rho1,Gamma1)
K2 = K1 * pow( rho1, Gamma1-Gamma2)
K3 = K2 * pow( rho2, Gamma2-Gamma3)


# Low-density part (PP approximation to low-density SLy EOS in [Read et al 2009](https://ui.adsabs.harvard.edu/#abs/2009PhRvD..79l4033R/abstract))

rhoL_1 = 2.62789e12/Density
rhoL_2 = 3.78358e11/Density
rhoL_3 = 2.44034e7/Density
rhoL_4 = 0.0

GammaL_1 = 1.35692
GammaL_2 = 0.62223
GammaL_3 = 1.28733
GammaL_4 = 1.58425

KL_1 = 3.99874e-8 * pow(Msun/Length**3, GammaL_1-1)  # notice a missing c^2 in Ki values in Table II of Read et al. 2009
KL_2 = 5.32697e+1 * pow(Msun/Length**3, GammaL_2-1) 
KL_3 = 1.06186e-6 * pow(Msun/Length**3, GammaL_3-1)  
KL_4 = 6.80110e-9 * pow(Msun/Length**3, GammaL_4-1)  

epsL_4 = 0.0
alphaL_4 = 0.0
epsL_3 = (1+alphaL_4)*rhoL_3 + KL_4/(GammaL_4 - 1)*pow(rhoL_3, GammaL_4)
alphaL_3 = epsL_3/rhoL_3 - 1 - KL_3/(GammaL_3 - 1)*pow(rhoL_3, GammaL_3 -1)
epsL_2 = (1+alphaL_3)*rhoL_2 + KL_3/(GammaL_3 - 1)*pow(rhoL_2, GammaL_3)
alphaL_2 = epsL_2/rhoL_2 - 1 - KL_2/(GammaL_2 - 1)*pow(rhoL_2, GammaL_2 -1)
epsL_1 = (1+alphaL_2)*rhoL_1 + KL_2/(GammaL_2 - 1)*pow(rhoL_1, GammaL_2)
alphaL_1 = epsL_1/rhoL_1 - 1 - KL_1/(GammaL_1 - 1)*pow(rhoL_1, GammaL_1 -1)

rho0 = pow(KL_1/K1,1.0/(Gamma1-GammaL_1))
eps0 = (1.0+alphaL_1)*rho0 + KL_1/(GammaL_1-1.0)*pow(rho0,GammaL_1)

alpha1 = eps0/rho0 - 1 - K1/(Gamma1 - 1)*pow(rho0, Gamma1 -1)
eps1 = (1+alpha1)*rho1 + K1/(Gamma1 - 1)*pow(rho1, Gamma1)
alpha2 = eps1/rho1 - 1 - K2/(Gamma2 - 1)*pow(rho1, Gamma2 -1)
eps2 = (1+alpha2)*rho2 + K2/(Gamma2 - 1)*pow(rho2, Gamma2)
alpha3 = eps2/rho2 - 1 - K3/(Gamma3 - 1)*pow(rho2, Gamma3 -1)

args = (rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2,KL_1,K1,K2,K3,        GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3)

def P_of_rho(rho, args):
    rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2,    KL_1,K1,K2,K3,GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3 = args
    if rho<rhoL_3:
        return KL_4*pow(rho,GammaL_4)
    elif rhoL_3<= rho <rhoL_2:
        return KL_3*pow(rho,GammaL_3)
    elif rhoL_2<= rho <rhoL_1:
        return KL_2*pow(rho,GammaL_2)
    elif rhoL_1<= rho <rho0:
        return KL_1*pow(rho,GammaL_1)
    elif rho0<= rho <rho1:
        return K1*pow(rho,Gamma1)
    elif rho1<= rho <rho2:
        return K2*pow(rho,Gamma2)
    else:
        return K3*pow(rho,Gamma3)

def rho_of_P(p, args):
    rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2, KL_1,K1,K2,K3,GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3 = args
    if p<pL_3:
        return pow(p/KL_4, 1.0/GammaL_4)
    elif pL_3<= p <pL_2:
        return pow(p/KL_3, 1.0/GammaL_3)
    elif pL_2<= p <pL_1:
        return pow(p/KL_2, 1.0/GammaL_2)
    elif pL_1<= p <p0:
        return pow(p/KL_1, 1.0/GammaL_1)
    elif p0<= p <p1:
        return pow(p/K1, 1.0/Gamma1)
    elif p1<= p <p2:
        return pow(p/K2, 1.0/Gamma2)
    else:
        return pow(p/K3, 1.0/Gamma3)


pL_3 = KL_3*pow(rhoL_3,GammaL_3)
pL_2 = KL_2*pow(rhoL_2,GammaL_2)
pL_1 = KL_1*pow(rhoL_1,GammaL_1)
p0 = KL_1*pow(rho0,GammaL_1)
p2 = K2*pow(rho2,Gamma2)
args2 = (rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2,KL_1,K1,K2,K3,GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3,pL_3,pL_2, pL_1, p0, p1, p2, alphaL_4, alphaL_3, alphaL_2, alphaL_1, alpha1, alpha2, alpha3)


def eps_of_rho(rho, args2):
    rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2,KL_1,K1,K2,K3,GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3,pL_3,pL_2, pL_1, p0, p1, p2,alphaL_4, alphaL_3, alphaL_2, alphaL_1, alpha1, alpha2, alpha3 = args2
    if rho<rhoL_3:
        return (1.0+alphaL_4)*rho + KL_4/(GammaL_4-1.0)*pow(rho,GammaL_4)
    elif rhoL_3<= rho <rhoL_2:
        return (1.0+alphaL_3)*rho + KL_3/(GammaL_3-1.0)*pow(rho,GammaL_3)
    elif rhoL_2<= rho <rhoL_1:
        return (1.0+alphaL_2)*rho + KL_2/(GammaL_2-1.0)*pow(rho,GammaL_2)
    elif rhoL_1<= rho <rho0:
        return (1.0+alphaL_1)*rho + KL_1/(GammaL_1-1.0)*pow(rho,GammaL_1)
    elif rho0<= rho <rho1:
        return (1.0+alpha1)*rho + K1/(Gamma1-1.0)*pow(rho,Gamma1)
    elif rho1<= rho <rho2:
        return (1.0+alpha2)*rho + K2/(Gamma2-1.0)*pow(rho,Gamma2)
    else:
        return (1.0+alpha3)*rho + K3/(Gamma3-1.0)*pow(rho,Gamma3)

def eps_of_P(p, args2):
    rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2,KL_1,K1,K2,K3,    GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3,    pL_3,pL_2, pL_1, p0, p1, p2,    alphaL_4, alphaL_3, alphaL_2, alphaL_1, alpha1, alpha2, alpha3 = args2    
    if p<pL_3:
        return (1.0+alphaL_4)*pow(p/KL_4, 1.0/GammaL_4)+ p/(GammaL_4-1)
    elif pL_3<= p <pL_2:
        return (1.0+alphaL_3)*pow(p/KL_3, 1.0/GammaL_3)+ p/(GammaL_3-1)
    elif pL_2<= p <pL_1:
        return (1.0+alphaL_2)*pow(p/KL_2, 1.0/GammaL_2)+ p/(GammaL_2-1)
    elif pL_1<= p <p0:
        return (1.0+alphaL_1)*pow(p/KL_1, 1.0/GammaL_1)+ p/(GammaL_1-1)
    elif p0<= p <p1:
        return (1.0+alpha1)*pow(p/K1, 1.0/Gamma1)+ p/(Gamma1-1)
    elif p1<= p <p2:
        return (1.0+alpha2)*pow(p/K2, 1.0/Gamma2)+ p/(Gamma2-1)
    else:
        return (1.0+alpha3)*pow(p/K3, 1.0/Gamma3)+ p/(Gamma3-1)


logrhopoints = np.arange(np.log10(1e5/Density),np.log10(10*rho2),0.1)
eospoints = len(logrhopoints)
logrhopointsCGS = logrhopoints + np.log10(Density)


logPpointsCGS = np.zeros(eospoints)
for i in range(0,eospoints):
    logPpointsCGS[i] = np.log10(Density*c**2*P_of_rho( pow(10.0,logrhopoints[i]), args))


plt.xlim(logrhopointsCGS[0], logrhopointsCGS[eospoints-1])
plt.ylim(20, 40)
plt.xlabel(r'$ \log_{10} (\rho \ {\rm in \ g/cm^3} )$')
plt.ylabel(r'$ \log_{10} (P \ {\rm in \ dyne/cm^2} )$')
plt.plot(logrhopointsCGS, logPpointsCGS)
xcoords = [np.log10(rhoL_4*Density), np.log10(rhoL_3*Density), np.log10(rhoL_2*Density), 
          np.log10(rhoL_1*Density), np.log10(rho0*Density), np.log10(rho1*Density), np.log10(rho2*Density)]
for xc in xcoords:
    plt.axvline(x=xc, color='black')


# ## Find the central energy density and pressure

eps_c = eps_of_rho(rho_c, args2)
P_c = P_of_rho(rho_c, args)


# ## Define the system of ODEs to be solved

def f(r, y, args2):
    rhoL_3,rhoL_2,rhoL_1,rho0,rho1,rho2,KL_4,KL_3,KL_2,KL_1,K1,K2,K3,GammaL_4,GammaL_3,GammaL_2,GammaL_1,Gamma1,Gamma2,Gamma3,pL_3,pL_2, pL_1, p0, p1, p2,alphaL_4, alphaL_3, alphaL_2, alphaL_1, alpha1, alpha2, alpha3 = args2
    
    eps = eps_of_P(y[0], args2) 
    
    return [ -( eps + y[0] )*( y[1] + 4.0*np.pi*pow(r,3.0)*y[0] )/( r*(r-2.0*y[1]) ), 
            
             4*np.pi*pow(r,2.0)*eps,
            
             2.0*( y[1] + 4.0*np.pi*pow(r,3.0)*y[0] )/( r*(r-2.0*y[1]) ) 
           ]


# ## Set the central value of $\nu_c$ and the starting values for the system of ODEs:

# set an arbitrary starting value for nu at the center
nu_c = -1.0

# set a safe max r, based on 4x radius of 1 Msun uniform density Newt. model
r_max = 4.0 * pow( 3.0/(4.0*np.pi*eps_c), 1.0/3.0)

# create an equidistant array of values for r
#Npoints = (51, 101, 201, 401, 801, 1601,  3201, 6401, 12801, 25601, 51201)
N = 2501
r = np.linspace(0.0, r_max, N)
dr = r[1] - r[0]

# compute P, m, nu at r=dr by Taylor expansion
P_1 =  P_c - (2.0*np.pi)*(eps_c+P_c)*(P_c+(1.0/3.0)*eps_c)*pow(dr,2.0)
m_1 =  (4.0/3.0)*np.pi*eps_c*pow(dr, 3.0)
nu_1 = nu_c + 4.0*np.pi*(P_c+(1.0/3.0)*eps_c)*pow(dr,2.0)

# set starting values at r=dr for numerical integration
y0 = [P_1, m_1, nu_1]


# ## Numerical solution

solve = integrate.ode(f)
solve.set_integrator('lsoda', rtol=1e-12, atol=1e-50,ixpr=True);
solve.set_initial_value(y0, dr);
solve.set_f_params(args2);


# Integrate from starting point to the surface (where $P=0$):

# create the solution vector
y = np.zeros((len(r), len(y0)))

# fill the solution vector with the values at the center
y[0,:] = [P_c, 0.0, nu_c]

# initialize counter
idx = 1

# integrate repeatedly to next grid point until P becomes zero
with stdout_redirected(): # suppress warnings
    while solve.successful() and solve.t < r[-1] and solve.y[0]>0.0:
        y[idx, :] = solve.y
        solve.integrate(solve.t + dr)
        idx += 1

# last grid point with positive pressure
idxlast = idx-1 

# radius at last positive pressure grid point
R_last = r[idxlast]

# mass at last positive pressure grid point
Mass_last = y[idxlast][1]


# Locate real radius by finding the location where h=1.0.

# use last 4 points to construct interpolant
r_data = np.zeros(4)
h_data = np.zeros(4)
eps_data = np.zeros(4)
rho_data = np.zeros(4)
P_data = np.zeros(4)
dmdr_data = np.zeros(4)

for i in range(idxlast-3,idxlast+1):
    r_data[i-idxlast+3] = r[i]
    eps_data[i-idxlast+3] = eps_of_P(y[i][0],args2)
    rho_data[i-idxlast+3] = rho_of_P(y[i][0],args)
    P_data[i-idxlast+3] = y[i][0]
    h_data[i-idxlast+3] = (eps_data[i-idxlast+3] + P_data[i-idxlast+3]) / rho_data[i-idxlast+3] -1.0
    dmdr_data[i-idxlast+3] = 4.0*np.pi*r[i]**2*eps_data[i-idxlast+3]

h_interp = PchipInterpolator(r_data, h_data)

Radius = optimize.brentq( h_interp, r_data[0], r_data[3]+3*dr, xtol=1e-16 )

# Locate radius more accurately (to 4th-order) using a cubic Hermite interpolant of the specific enthalpy h-1.

def hHerm (r):
    r_last_1 = R_last-dr
    r_last = R_last
    w = (r-r_data[2])/dr
    m_last_1 = y[idxlast-1][1]
    m_last = y[idxlast][1]
    dhdr_last_1 = - (h_data[2]+1.0)*(m_last_1 + 4.0*np.pi*r_last_1**3*y[idxlast-1][0])/ (r_last_1*(r_last_1-2.0*m_last_1))
    dhdr_last = - (h_data[3]+1.0)*(m_last + 4.0*np.pi*r_last**3*y[idxlast][0])/(r_last*(r_last-2.0*m_last))
    return (h_data[2]+1.0)*(2.0*pow(w,3.0)-3.0*pow(w,2.0)+1.0)+(h_data[3]+1.0)*(2.0*pow(1.0-w,3.0)-3.0*pow(1.0-w,2.0)+1.0) + ( dhdr_last_1*(pow(w,3.0)-2.0*pow(w,2.0)+w) - dhdr_last*(pow(1-w,3.0)-2.0*pow(1-w,2.0)+1-w))*dr -1.0


Radius = optimize.brentq( hHerm, r_data[0], r_data[3]+3*dr, xtol=1e-16 )


# Correct mass by adding last missing piece by Simpson's rule (finding an intemediate point by pchip interpolation):

dmdr_interp_pchip = PchipInterpolator(r_data, dmdr_data)
dmdr_midpoint = dmdr_interp_pchip((R_last+Radius)/2)
Dmass_simps = (1.0/3.0)*(Radius-R_last)/2*(dmdr_interp_pchip(R_last)+4.0*dmdr_midpoint+dmdr_interp_pchip(Radius))

Mass = Mass_last + Dmass_simps

# Construct table with main solution variables:

values = np.zeros((idxlast+1, 10)) 

for i in range(0,idxlast+1): 
    values[i][0] = r[i]
    values[i][1] = rho_of_P(y[i][0], args) # rho
    values[i][2] = eps_of_P(y[i][0], args2) # epsilon
    values[i][3] = y[i][0]   # P
    values[i][4] = y[i][1]   # m
    values[i][5] = y[i][2]   # nu (arbitrary)

values[0][6] = 0.0
for i in range(1,idxlast+1):     
    values[i][6] = - np.log(1.0-2.0*y[i][1]/r[i])   # lambda
    
values[:, 7] = (values[:, 2] + values[:, 3])/values[:, 1]  # h

values[:, 8] = - (values[:, 4] + 4.0*np.pi*pow(values[:, 0], 3.0)*values[:, 3])/ ( values[:, 0]*(values[:, 0] - 2.0*values[:, 4]))
                    # (e+P)^{-1} dP/dr directly from rhs of TOV eqn
        
values[0][8] = 0.0   # fix value at the center 

for i in range(0,idxlast+1):
    rho = values[i][1] 
    if rho<rhoL_3:
        values[i][9] = GammaL_4
    elif rhoL_3<= rho <rhoL_2:
        values[i][9] = GammaL_3
    elif rhoL_2<= rho <rhoL_1:
        values[i][9] = GammaL_2
    elif rhoL_1<= rho <rho0:
        values[i][9] = GammaL_1
    elif rho0<= rho <rho1:
        values[i][9] = Gamma1
    elif rho1<= rho <rho2:
        values[i][9] = Gamma2
    else:
        values[i][9] = Gamma3


# Match $\nu$ at the surface, using Schwarzshild vacuum solution:

# arbitrary nu at the surface
nu_s_old = y[idxlast][2]

# correct nu at the surface
nu_s = np.log(1.0-2.0*Mass/Radius)

# shift nu inside star by difference
values[:, 5] = values[:, 5] + (-nu_s_old + nu_s)


# Compute baryon mass and alternative expression for gravitational mass:

# construct radius array and integrands for baryon and alternative mass integration

rint = np.zeros(idxlast+1)
m0int = np.zeros(idxlast+1)
mint_alt = np.zeros(idxlast+1)

# fill radius array and integrands 

for i in range(0,idxlast+1): 
    rint[i] = values[i][0]
    m0int[i] = 4.0*np.pi*pow(rint[i],2.0)*np.exp(values[i][6]/2.0)*values[i][1]
    mint_alt[i] = 4.0*np.pi*pow(rint[i],2.0)*np.exp((values[i][5]+values[i][6])/2.0)*(values[i][2]+3.0*values[i][3])

# integrate using Simpson's method
M0_last = integrate.simps( m0int, dx=dr)
M_alt_last = integrate.simps( mint_alt, dx=dr, even='last')

# correct M0 and M_alt by adding last trapezoid
M0 = M0_last + 0.5*4.0*np.pi*R_last**2*np.exp(values[idxlast][6]/2.0)*values[idxlast][1]*(Radius-R_last)

M_alt = M_alt_last + 0.5*4.0*np.pi*R_last**2* np.exp((values[idxlast][5]
                            +values[idxlast][6])/2.0)*(values[idxlast][2] \
                                +3.0*values[idxlast][3]) *(Radius-R_last)

# compute relative difference between mass and alt. mass
M_reldiff = (Mass-M_alt)/Mass


# # Main results

N_gridpoints = idxlast+1

# SCREEN OUTPUT 

print("%5.4e %5.4e %12.11f %12.11f %12.11g %d" % (rho_c*Density, eps_c*Density, Mass, M0, Radius*Length/1e5, N_gridpoints))

# Convert to CGS

values_CGS = np.zeros((idxlast+1, 10)) 

values_CGS[:, 0] = values[:, 0] * Length
values_CGS[:, 1] = values[:, 1] * Density  # rho
values_CGS[:, 2] = values[:, 2] * Density*c**2  # epsilon
values_CGS[:, 3] = values[:, 3] * Density*c**2  # P
values_CGS[:, 4] = values[:, 4] * Msun  # m
values_CGS[:, 5] = values[:, 5]         # nu
values_CGS[:, 6] = values[:, 6]         # lambda
values_CGS[:, 7] = values[:, 7] * c**2  # h
values_CGS[:, 8] = values[:, 8] / Length   # (epsilon+P)^{-1} dP/dr
values_CGS[:, 9] = values[:, 9]         # Gamma


# Write output files

np.savetxt('TOV_output.dat', values)
np.savetxt('TOV_output_CGS.dat', values_CGS)


