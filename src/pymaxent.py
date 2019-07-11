#!/usr/bin/env python
"""PyMaxEnt.py: Implements a maximum entropy reconstruction of distributions with known moments."""

__author__     = "Tony Saad and Giovanna Ruai"
__copyright__  = "Copyright (c) 2019, Tony Saad"

__credits__    = ["University of Utah Department of Chemical Engineering", "University of Utah UROP office"]
__license__    = "MIT"
__version__    = "1.0.0"
__maintainer__ = "Tony Saad"
__email__      = "tony.saad@chemeng.utah.edu"
__status__     = "Production"

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

def moments_c(f, k=0, bnds=[-np.inf, np.inf]):
    '''
    Creates "k" moments: μ0, μ1, ..., μ(k-1) for a function "f" on the support given by "bnds".

    -Inupts-    
    f: distribution function ##must be in the form of a function##
    bnd: a list of two numbers consisting of the lower and upper bounds of the support    
    k: integer number of moments to compute. Will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)
    bnds: boundaries for the integration
    -Outputs-
    moments: an array of moments of length "k"    

    -Example-
    μ = moments(3, f, [-1, 1])    
    '''
    def mom(x, k):
        return x**k*f(x)
    
    moms = np.zeros(k)
    a = bnds[0]
    b = bnds[1]
    for i in range(0,k):
        moms[i] = quad(mom,a,b,args = i)[0]
    return moms

def moments_d(f,k,x):
    '''
    Calculates the first "k" moments: μ0, μ1, ..., μ(k-1) of a discrete distribution "f".
    -Inputs-
    f: an array of values for a discrete distribution
    k: integer, number of moments to compute. Will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)
    x: list or array of independent variables
    -Outputs-
    mom: an array of length k containing the moments for the known distribution
    '''
    moms = []
    for i in range(0,k):
        xp = np.power(x,i)      # compute x^p
        xpf = np.dot(xp,f)      # compute x^p * f(x)
        mom.append(np.sum(xpf)) # compute moment: sum(x^p * f(x))
    return np.array(moms)

def moments(f, k, ivars=None, bnds=None):
    '''
    Computes the first "k" moments of a function "f" on the support given by "bnd". If "ivars" is provided, then a discrete distribution is assumed and "f" ##must## be a list or array of scalar values.

    -Inupts-
    f: distribution function ##must be in the form of a function##
    k: will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)
    ivars: optional - designates a list or array of discrete values for an independent variable. If x is provided, then the moments will be computed based on a discrete distribution. This means that f must be an array as well.
    bnds: a list of two numbers consisting of the lower and upper bounds of the support    
    
    -Outputs-
    moments: an array of moments of length "k"    

    -Example-
    μ = moments(3, f, [-1, 1])    
    '''    
    if ivars is not None:
        if bnds is not None:
            print('WARNING: You specified BOTH x and boundaries. I will assume this is a discrete distribution. If you want to calculate a continuous distribution, please specify bnd ONLY.')
        return moments_d(f,k,ivars)
    else:
        return moments_c(f,k,bnds)

def integrand(x, lamb, m, k=0, discrete=False):
    '''
    Calculates the integrand for the integral found in the moment approximation function.
    -Inputs-
    x: support for the distribution
    lamb: an array of Lagrange constants used to approximate the distribution
    k: a constant representing the order of the moment being approximated 
    -Outputs-
    integrand: the caclulated portion of the integrand at each x value
    '''
    neqs = len(lamb)
    xi = np.array([x**i for i in range(0, neqs)])
    if discrete:
        return m * x**k * np.exp(np.dot(lamb, xi))
    else:
        return m(x) * x**k * np.exp(np.dot(lamb, xi))

    
# def discrete_eq(lamb,x,k,m=1):
#     '''
#     Calculates an approximated moment of order "k"
#     -Inputs-
#     lamb: an array of constants used to approximate a probability
#     x: support
#     '''
#     neqs = len(lamb)
#     xi = np.array([x**i for i in range(0, neqs)])
#     result = m * x**k * np.exp(np.dot(lamb, xi))
#     return result



def residual_d(lamb,x,k,mu,m=1):
    '''
    Calculates the residual of the moment approximation function.
    -Inputs-
    lamb: an array of Lagrange constants used to approximate the distribution
    mu: an array of the known moments needed to approximate the distribution function
    bnd: support bounds
    -Outputs-
    rhs: the integrated right hand side of the moment approximation function
    '''
    l_sum = []
    for i in range(0,len(lamb)):
        l_sum.append( np.sum(integrand(x,lamb,m,i,discrete=True)) - mu[i] )
#         l_sum.append(np.sum(discrete_eq(lamb,x,i,m))-mu[i])            
    return np.array(l_sum)

def maxent_reconstruct_d(x, mu, m):
    '''
    Computes the most likely distribution from the moments given using maximum entropy theorum.
    -Inputs-
    mu: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
        Ex. μ = [1,0,0]
    bnd: Support for the distribution [a,b]
        ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    -Outputs-
    probabilites: vector of size b (from bnd[1]) containing the probabilities for the distribution 
    '''
    lambguess = np.zeros(len(mu))
    lambguess[0] = -np.log(np.sqrt(2*np.pi))
    k = len(mu)
    lambsol = fsolve(residual_d, lambguess, args = (x,k,mu,m))
#     probabilites = discrete_eq(lambsol,x,k=0,m)
    probabilites = integrand(x,lambsol,m,k=0,discrete=True)    
    return probabilites, lambsol


def residual_c(lamb, mu, bnd, m):
    '''
    Calculates the residual of the moment approximation function.
    -Inputs-
    lamb: an array of Lagrange constants used to approximate the distribution
    mu: an array of the known moments needed to approximate the distribution function
    bnd: support bounds
    -Outputs-
    rhs: the integrated right hand side of the moment approximation function
    '''
    a = bnd[0]
    b = bnd[1]
    neqs = len(lamb)
    rhs = np.zeros(neqs)
    for k in range(0, neqs):
        rhs[k] = quad(integrand, a, b, args=(lamb,m,k))[0] - mu[k]
    return rhs

def maxent_reconstruct_c(mu, bnd=[-np.inf, np.inf], m=None):
    '''
    Used to construct a continuous distribution from a limited number of known moments(μ). This function applies Maximum Entropy Theory in order to solve for the constraints found in the approximation equation that is given as an output.
    -Inputs-    
    μ: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
        Ex. μ = [1,0,0]
    bnd: Support for the integration [a,b]
        ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    -Outputs-
    Distribution Function: The recreated probability distribution function from the moment vector (μ) input given. requires a support to be ploted
        Ex. f, sol = maxent([1,0,0], [-1,1])
            ## f = lambda x: integrand(x,sol,0) ##
            x_axis = some vector to evaluate f by
            f(x_axis) = vector of probability distribution values        
    sol: Solution to the lagrange constants found in the moment approximation function (see fxn integrand) in the form of a vector. sol = [λ_1 , λ_2 , ...]
    '''
    neqs = len(mu)
    lambguess = np.zeros(neqs) # initialize guesses
    lambguess[0] = -np.log(np.sqrt(2*np.pi)) # set the first initial guess - this seems to work okay
    lambsol = fsolve(residual_c, lambguess, args=(mu,bnd, m), col_deriv=True)
    return lambda x: integrand(x, lambsol, m,k=0), lambsol

def reconstruct(mu, ivars=None, bnds=None, scaling=None):
    '''
    This is the main function call to generate maximum entropy solutions.
    mu: a list or array of known moments
    ivars: optional - a list or array of known dependent variables. For example, for a 6-faced die, ivars=[1,2,3,4,5,6]. If ivars is provided, we will assume a discrete reconstruction.
    bnds: a tuple [a,b] containing the bounds or support of the reconstructed solution. This is only required for continuous distributions and will be neglected if ivars is provided.
    scaling: this is the invariant measure or scaling function. If ivars is provided (i.e. a discrete distribution reconstruction), then scaling ##must## be a list or an array of scalar values.
    '''
    result = 0
    # Discrete case
    if ivars is not None:
        ivars = np.array(ivars) # convert things to numpy arrays
        if bnds is not None:
            print('WARNING: You specified BOTH x and boundaries. I will assume this is a discrete distribution. If you want to calculate a continuous distribution, please specify bnd ONLY.')
        if scaling is None:
            # if the invariant measure is not provided, then assume it is one
            scaling = np.ones(len(ivars))
        scaling = np.array(scaling)
        result = maxent_reconstruct_d(ivars, mu, scaling)
    # Continuous case
    else:
        if scaling is None:
            scaling = lambda x: 1
        result = maxent_reconstruct_c(mu, bnds, scaling)
    return result
