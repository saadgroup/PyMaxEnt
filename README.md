# PyMaxEnt
`PyMaxEnt` is a python software that solves the inverse moment problem using the maximum entropy principle. Given a finite number of moments, PyMaxEnt will find the function that generates those moments and also maximizes the information entropy.

# Usage
To use `PyMaxEnt`, a single function call to `reconstruct(moments, ivars, bnds, scaling)` is made. Here, `moments` is a required list or array of known moments, `ivars` is an optional argument containing discrete values of the independent variable, `bnds` is a tuple `[a,b]` containing the expected bounds of the resulting distribution, and `scaling` is the invariant measure or the scaling function. 

When `ivars` is provided, the reconstruction assumes a discrete distribution. When a discrete reconstruction is chosen, `scaling` should be an array of the same size as `ivars`.

More details can be found in `src/pymaxent.py`.

## Sample code snippets

Below are some examples of using the \MaxEnt software. Starting with an example to reconstruct a basic dicrete distribution with two moments and four independent variables

```python
from pymaxent import *
mu = [1,3.5]
x = [1,2,3,4,5,6]
sol, lambdas = reconstruct(mu,ivars=x)
```

To scale the distribution, we simply pass a scaling function
```python
from pymaxent import *
mu = [1,2.5]
x = [1,2,3,8]
f = [1,2,3,4]
sol, lambdas = reconstruct(mu,x=x,scaling=f)
```

Similarly, for a continuous distribution, one passes a list of input moments. 
In this case, however, one must specify the bounds (`bnds`) to indicate that this is a continuous reconstruction. 
Here's an example for a Gaussian distribution
```python
from pymaxent import *
mu = [1,0,0.04]
sol, lambdas = reconstruct(mu,bnds=[-1,1])
# plot the reconstructed solution
x = np.linspace(-1,1)
plot(x,sol(x))
```
To scale the Gaussian, simply pass a function to the scaling argument
```python
from pymaxent import *
mu = [1,0,0.04]
f = lambda x: x**2
sol, lambdas = reconstruct(mu, bnds=[-1,1], scaling=f)
# plot the reconstructed solution
x = np.linspace(-1,1)
plot(x,sol(x))
```
