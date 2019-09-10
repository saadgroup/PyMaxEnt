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

    Parameters:
        f (function): distribution function **must be in the form of a function**
        k (int): integer number of moments to compute. Will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)
        bnds (tuple): boundaries for the integration

    Returns:
        moments: an array of moments of length "k"
    
    Example:
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
    
    Parameters:
        f (array): an array of values for a discrete distribution        
        k (int): number of moments to compute. Will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)        
        x (array): list or array containing the values of the random variable over which the distribution is to be integrated

    Returns:
        mom: an array of length k containing the moments for the known distribution
    '''
    moms = []
    for i in range(0,k):
        xp = np.power(x,i)      # compute x^p
        xpf = np.dot(xp,f)      # compute x^p * f(x)
        mom.append(np.sum(xpf)) # compute moment: sum(x^p * f(x))
    return np.array(moms)

def moments(f, k, rndvar=None, bnds=None):
    '''
    Computes the first "k" moments of a function "f" on the support given by "bnd". If "rndvar" is provided, then a discrete distribution is assumed and "f" ##must## be a list or array of scalar values.

    Parameters:
        f (function): distribution function **must be in the form of a function**
        k (integer): will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)
        rndvar (array): optional - designates a list or array of discrete values for a random variable. If x is provided, then the moments will be computed based on a discrete distribution. This means that f must be an array as well.
        bnds (tuple): a list of two numbers consisting of the lower and upper bounds of the support    
    
    Returns:
        moments: an array of moments of length `k`    

    Example:
        μ = moments(3, f, [-1, 1])    
    '''    
    if rndvar is not None:
        if bnds is not None:
            print('WARNING: You specified BOTH x and boundaries. I will assume this is a discrete distribution. If you want to calculate a continuous distribution, please specify bnd ONLY.')
        return moments_d(f,k,rndvar)
    else:
        return moments_c(f,k,bnds)

def integrand(x, lamb, m, k=0, discrete=False):
    '''
    Calculates the integrand of the \(k^\mathrm{th}\) moment.

    Parameters:
        x (array): linear space or set of values for a random variable on which the integrand is applied
        lamb (array): an array of Lagrange multipliers used to approximate the distribution
        m: invariant measure or scaling function. if `discrete` is True, then `m` **must be an array**, otherwise, `m` is a generic python function of the form `m(x)`
        k (integer): a constant representing the order of the moment being calculated

    Returns:
        integrand: the caclulated portion of the integrand at each x value
    '''
    neqs = len(lamb)
    xi = np.array([x**i for i in range(0, neqs)])
    if discrete:
        return m * x**k * np.exp(np.dot(lamb, xi))
    else:
        return m(x) * x**k * np.exp(np.dot(lamb, xi))

def residual_d(lamb,x,k,mu,m=1):
    '''
    Calculates the residual of the moment approximation function.

    Parameters:
        lamb (array): an array of Lagrange constants used to approximate the distribution
        x (array): 
        k (integer): order of the moment        
        mu (array): an array of the known moments needed to approximate the distribution function
        m: invariant measure
    
    Returns:
        rhs: the integrated right hand side of the moment approximation function
    '''
    l_sum = []
    for i in range(0,len(lamb)):
        l_sum.append( np.sum(integrand(x,lamb,m,i,discrete=True)) - mu[i] )
    return np.array(l_sum)

def maxent_reconstruct_d(rndvar, mu, m):
    '''
    Computes the most likely distribution from the moments given using maximum entropy theorum.

    Parameters:
        rndvar (array): a list or array of known dependent variables. For example, for a 6-faced die, rndvar=[1,2,3,4,5,6]
        mu (array): vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc... For example, μ = [1,0,0]
        m (array): invariant measure or scaling function. **Must be an array of the same size as `rndvar`

    Returns:
        probabilites: vector of size b (from bnd[1]) containing the probabilities for the distribution 
        lambsol: vector of lagrangian multipliers
    '''
    lambguess = np.zeros(len(mu))
    lambguess[0] = -np.log(np.sqrt(2*np.pi))
    k = len(mu)
    lambsol = fsolve(residual_d, lambguess, args = (rndvar,k,mu,m))
    probabilites = integrand(rndvar, lambsol, m, k=0, discrete=True)    
    return probabilites, lambsol


def residual_c(lamb, mu, bnds, m):
    '''
    Calculates the residual of the moment approximation function.
    
    Parameters:
        lamb (array): an array of Lagrange constants used to approximate the distribution
        mu (array): an array of the known moments needed to approximate the distribution function
        bnds (tuple): support bounds
        m (function): invariant measure or scaling function `m(x)`

    Returns:
        rhs: the integrated right hand side of the moment approximation function
    '''
    a = bnds[0]
    b = bnds[1]
    neqs = len(lamb)
    rhs = np.zeros(neqs)
    for k in range(0, neqs):
        rhs[k] = quad(integrand, a, b, args=(lamb,m,k))[0] - mu[k]
    return rhs

def maxent_reconstruct_c(mu, bnds=[-np.inf, np.inf], m=None):
    '''
    Used to construct a continuous distribution from a limited number of known moments(μ). This function applies Maximum Entropy Theory in order to solve for the constraints found in the approximation equation that is given as an output.
    
    Parameters:
        μ: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
            Ex. μ = [1,0,0]
        bnds: Support for the integration [a,b]
            ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    Returns:
        Distribution Function: The recreated probability distribution function from the moment vector (μ) input given. requires a support to be ploted
    
    Example:
        >>> f, sol = maxent([1,0,0], [-1,1])        
    '''
    neqs = len(mu)
    lambguess = np.zeros(neqs) # initialize guesses
    lambguess[0] = -np.log(np.sqrt(2*np.pi)) # set the first initial guess - this seems to work okay
    lambsol = fsolve(residual_c, lambguess, args=(mu,bnds, m), col_deriv=True)
    recon = lambda x: integrand(x, lambsol, m, k=0)
    return recon, lambsol

def reconstruct(mu, rndvar=None, bnds=None, scaling=None):
    '''
    This is the main function call to generate maximum entropy solutions.
    
    Parameters:
        mu (array): a list or array of known moments
        rndvar (array): optional - a list or array of known dependent variables. For example, for a 6-faced die, rndvar=[1,2,3,4,5,6]. If rndvar is provided, we will assume a discrete reconstruction.
        bnds (tuple): a tuple [a,b] containing the bounds or support of the reconstructed solution. This is only required for continuous distributions and will be neglected if rndvar is provided.
        scaling (function/array): this is the invariant measure or scaling function. If rndvar is provided (i.e. a discrete distribution reconstruction), then scaling ##must## be a list or an array of scalar values.
    
    Returns:
        recon: reconstructed distribution. If continuous, then `recon` is a Python function, `f(x)`. If discrete, then recon is an array of probabilities.
        lambsol (array): array containing the lagrangian multipliers
    
    Examples:
        ### reconstruct a discrete distribution
        >>> from pymaxent import *
        >>> mu = [1,3.5]
        >>> x = [1,2,3,4,5,6]
        >>> sol, lambdas = reconstruct(mu,rndvar=x)

        ### reconstruct a scaled distribution
        >>> from pymaxent import *
        >>> mu = [1,2.5]
        >>> x = [1,2,3,8]
        >>> f = [1,2,3,4]
        >>> sol, lambdas = reconstruct(mu,x=x,scaling=f)
        
        ### reconstruct a continuous distribution
        >>> from pymaxent import *
        >>> mu = [1,0,0.04]
        >>> sol, lambdas = reconstruct(mu,bnds=[-1,1])
        >>> x = np.linspace(-1,1)
        >>> plot(x,sol(x))      
        
        ### reconstruct a scaled continuous distribution
        >>> from pymaxent import *
        >>> mu = [1,0,0.04]
        >>> f = lambda x: x**2
        >>> sol, lambdas = reconstruct(mu, bnds=[-1,1], scaling=f)        
        >>> x = np.linspace(-1,1)
        >>> plot(x,sol(x))
    '''
    result = 0
    # Discrete case
    if rndvar is not None:
        rndvar = np.array(rndvar) # convert things to numpy arrays
        if bnds is not None:
            print('WARNING: You specified BOTH x and boundaries. I will assume this is a discrete distribution. If you want to calculate a continuous distribution, please specify bnd ONLY.')
        if scaling is None:
            # if the invariant measure is not provided, then assume it is one
            scaling = np.ones(len(rndvar))
        scaling = np.array(scaling)
        result = maxent_reconstruct_d(rndvar, mu, scaling)
    # Continuous case
    else:
        if scaling is None:
            scaling = lambda x: 1
        result = maxent_reconstruct_c(mu, bnds, scaling)
    return result
