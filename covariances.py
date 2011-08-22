import math
from numpy import *
import numpy.ma as ma

def genSECovar(marginal_variance, lengthscale):
    return lambda x,y: marginal_variance * math.exp( -1 * linalg.norm(x-y)**2 / lengthscale )

def allpairs_covar(x, missing_obs, scenario):
    """
    Return the matrix giving covariances under a GP model between the residuals for all event-station pairs for which we have observations.

    Arguments:
      x is a list of locations of the events
      missing_obs is a boolean n*m matrix indicating which event-station pairs are missing
      scenario is a scenario object, specifying the station locations and the GP covariance.
      """

    n = len(x)
    m = len(scenario.stations)
  
    missing_obs = matrix(missing_obs)

    sn = m*n - count_true(missing_obs);
    sigma = zeros((sn, sn));

    c1 = -1;
    for s1 in range(m):
        for e1 in range(n):
            if missing_obs[e1,s1]:
                continue
            c1 = c1 + 1;

            c2 = -1;
            for s2 in range(m):
                for e2 in range(n):
                    if missing_obs[e2,s2]:
                        continue
                    c2 = c2 + 1;
                
                    if s2 < s1:
                        continue
                    
                    sigma[c1,c2] = scenario.calc_residual_covariance(s1, s2, x[e1,:], x[e2,:])
                    sigma[c2,c1] = sigma[c1,c2];

    return sigma
               
def count_true(a):
    return sum(map(lambda b: 1 if b else 0, a.flat))
