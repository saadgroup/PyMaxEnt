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
from scipy.optimize import root

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
        return x**k * f(x)

    moms = np.zeros(k)
    a = bnds[0]
    b = bnds[1]
    for i in range(0, k):
        moms[i] = quad(mom, a, b, args=i)[0]
    return moms

def moments_d(f, k, x):
    '''
    Calculates the first "k" moments: μ0, μ1, ..., μ(k-1) of a discrete distribution "f".
    
    Parameters:
        f (array): an array of values for a discrete distribution        
        k (int): number of moments to compute. Will evaluate the first k moments of f, μ0, μ1, ..., μ(k-1)        
        x (array): list or array containing the values of the random variable over which the distribution is to be integrated

    Returns:
        moms: an array of length k containing the moments for the known distribution
    '''
    x = np.asarray(x)
    f = np.asarray(f)
    moms = []
    for i in range(0, k):
        xp  = np.power(x, i)       # compute x^i
        xpf = np.dot(xp, f)        # compute sum(x^i * f(x))
        moms.append(xpf)            # fix: was mom.append (NameError in original)
    return np.array(moms)

def moments(f, k, rndvar=None, bnds=None):
    '''
    Computes the first "k" moments of a function "f" on the support given by "bnd". If "rndvar" is provided, then a discrete distribution is assumed and "f" must be a list or array of scalar values.

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
        return moments_d(f, k, rndvar)
    else:
        return moments_c(f, k, bnds)

def integrand(x, lamb, k=0, discrete=False):
    '''
    Calculates the integrand of the k-th moment.

    Parameters:
        x (array): linear space or set of values for a random variable on which the integrand is applied
        lamb (array): an array of Lagrange multipliers used to approximate the distribution
        k (integer): a constant representing the order of the moment being calculated
        discrete (bool): unused, kept for backwards compatibility

    Returns:
        integrand: the calculated portion of the integrand at each x value
    '''
    neqs = len(lamb)
    xi = np.array([x**i for i in range(0, neqs)])
    return x**k * np.exp(np.dot(lamb, xi))

def residual_d(lamb, x, k, mu):
    '''
    Calculates the residual of the moment approximation function for the discrete case.

    Parameters:
        lamb (array): an array of Lagrange constants used to approximate the distribution
        x (array): values of the random variable
        k (integer): number of moments
        mu (array): an array of the known moments needed to approximate the distribution function
    
    Returns:
        rhs: residual vector of length k
    '''
    # Build Vandermonde-style powers matrix: shape (k, len(x))
    # X[i,j] = x[j]^i
    X   = np.vstack([x**i for i in range(k)])   # (k, n)
    phi = np.exp(lamb @ X)                        # (n,)  -- partition fn, computed once
    rhs = X @ phi - np.asarray(mu)               # (k,)  -- fully vectorized
    return rhs

def maxent_reconstruct_d(rndvar, mu):
    '''
    Computes the most likely distribution from the moments given using maximum entropy theorem.

    Parameters:
        rndvar (array): a list or array of known dependent variables. For example, for a 6-faced die, rndvar=[1,2,3,4,5,6]
        mu (array): vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc... For example, μ = [1,0,0]

    Returns:
        probabilities: vector containing the probabilities for the distribution 
        lambsol: vector of Lagrangian multipliers
    '''
    mu       = np.asarray(mu)
    k        = len(mu)
    lambguess        = np.zeros(k)
    lambguess[0]     = -np.log(np.sqrt(2 * np.pi))

    sol = root(residual_d, lambguess, args=(rndvar, k, mu), method='hybr')
    lambsol       = sol.x
    probabilities = integrand(rndvar, lambsol, k=0)
    return probabilities, lambsol


def residual_c(lamb, mu, bnds):
    '''
    Calculates the residual of the moment approximation function for the continuous case.
    
    Parameters:
        lamb (array): an array of Lagrange constants used to approximate the distribution
        mu (array): an array of the known moments needed to approximate the distribution function
        bnds (tuple): support bounds

    Returns:
        rhs: residual vector of length neqs
    '''
    a    = bnds[0]
    b    = bnds[1]
    neqs = len(lamb)
    rhs  = np.zeros(neqs)
    for k in range(neqs):
        rhs[k] = quad(integrand, a, b, args=(lamb, k))[0] - mu[k]
    return rhs

def jacobian_c(lamb, mu, bnds):
    '''
    Analytical Jacobian of residual_c.  J[i,j] = integral of x^(i+j) * exp(lambda . xi) dx,
    which is the (i+j)-th moment of the MaxEnt distribution at the current lambda.

    Parameters:
        lamb (array): current Lagrange multipliers
        mu (array): known moments (not used in computation, kept for signature compatibility with root)
        bnds (tuple): support bounds

    Returns:
        J: (neqs x neqs) Jacobian matrix
    '''
    a    = bnds[0]
    b    = bnds[1]
    neqs = len(lamb)
    J    = np.zeros((neqs, neqs))
    for i in range(neqs):
        for j in range(neqs):
            J[i, j] = quad(integrand, a, b, args=(lamb, i + j))[0]
    return J

def maxent_reconstruct_c(mu, bnds=[-np.inf, np.inf]):
    '''
    Used to construct a continuous distribution from a limited number of known moments (μ).
    This function applies Maximum Entropy Theory in order to solve for the constraints found
    in the approximation equation that is given as an output.
    
    Parameters:
        mu: vector of size m containing the known moments of a distribution. This does NOT assume that μ0 = 1. This vector contains moments μ_k starting with μ_0, μ_1, etc...
            Ex. μ = [1,0,0]
        bnds: Support for the integration [a,b]
            ## It is important the bounds include roughly all non-zero values of the distribution that is being recreated ##
    
    Returns:
        recon: the recreated probability distribution function from the moment vector (μ) input given, as a callable f(x)
        lambsol: array of Lagrangian multipliers
    
    Example:
        >>> f, sol = maxent([1,0,0], [-1,1])        
    '''
    mu               = np.asarray(mu)
    neqs             = len(mu)
    lambguess        = np.zeros(neqs)
    lambguess[0]     = -np.log(np.sqrt(2 * np.pi))

    sol     = root(residual_c, lambguess, jac=jacobian_c, args=(mu, bnds), method='hybr')
    lambsol = sol.x
    recon   = lambda x: integrand(x, lambsol, k=0)
    return recon, lambsol

def reconstruct(mu, rndvar=None, bnds=None):
    '''
    This is the main function call to generate maximum entropy solutions.
    
    Parameters:
        mu (array): a list or array of known moments
        rndvar (array): optional - a list or array of known dependent variables. For example, for a 6-faced die, rndvar=[1,2,3,4,5,6]. If rndvar is provided, we will assume a discrete reconstruction.
        bnds (tuple): a tuple [a,b] containing the bounds or support of the reconstructed solution. This is only required for continuous distributions and will be neglected if rndvar is provided.
    
    Returns:
        recon: reconstructed distribution. If continuous, then `recon` is a Python function, `f(x)`. If discrete, then recon is an array of probabilities.
        lambsol (array): array containing the Lagrangian multipliers
    
    Examples:
        ### reconstruct a discrete distribution
        >>> from pymaxent import *
        >>> mu = [1,3.5]
        >>> x = [1,2,3,4,5,6]
        >>> sol, lambdas = reconstruct(mu,rndvar=x)
        
        ### reconstruct a continuous distribution
        >>> from pymaxent import *
        >>> mu = [1,0,0.04]
        >>> sol, lambdas = reconstruct(mu,bnds=[-1,1])
        >>> x = np.linspace(-1,1)
        >>> plot(x,sol(x))              
    '''
    # Discrete case
    if rndvar is not None:
        rndvar = np.asarray(rndvar)
        if bnds is not None:
            print('WARNING: You specified BOTH x and boundaries. I will assume this is a discrete distribution. If you want to calculate a continuous distribution, please specify bnd ONLY.')
        return maxent_reconstruct_d(rndvar, mu)
    # Continuous case
    else:
        return maxent_reconstruct_c(mu, bnds)
