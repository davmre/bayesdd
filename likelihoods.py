from numpy import *
import math
import covariances

def gp_likelihood(obs, events, scenario, sigma=None, temperature=1):
    """
    Compute the (log) likelihood of observing a particular set of arrival times, given a set of event locations and a scenario description (station locations and model parameters) defining a Gaussian process prior on the slowness field.

    Arguments:
      obs is an observation object containing the picked arrival times
      events is a list of event locations
      sc is a scenario object
      sigma is a matrix giving precomputed covariances between residuals for all event/station pairs, if available
      temperature is the temperature
    """
    n = len(events)
    m = len(scenario.stations)
    
    x = events[:,:-1]
    r = events[:,-1]

    # if covariances are not given, compute them
    if sigma is None:
        sigma = covariances.allpairs_covar(x, isnan(obs.picks), scenario)

    # pad the variance of each observation to account for pick error
    sn = sigma.shape[0]
    sigma = sigma + eye(sn) * (scenario.pick_stddev**2)

    # make sure the covariance matrix is positive semi-definite
    (v,d) = linalg.eig(sigma)
    if v.min() < 0:
        print("correcting allpairs by " + str(v.min()) + "*2.")
        sigma = sigma - eye(sn)*v.min()*2

    # calculate the difference between observed and predicted (mean) arrival times
    a = reshape(obs.picks, n*m)
    pair_origin_times = reshape(tile(r, (1, m)), n*m)
    tt = scenario.calc_traveltime(x, scenario.stations)
    pair_travel_times = reshape(tt, n*m)
    deviation = array(a - pair_origin_times - pair_travel_times)

    deviation = deviation[~isnan(deviation)]
    if len(deviation) != sn:
        print(sigma.shape)
        print(deviation.shape)
        raise Exception('covariance matrix should be the same size as deviation vector')

    # compute the likelihood
    log_likelihood = .5 * (n*m*log(2*math.pi) + math.log ( linalg.det(sigma) ) + dot(deviation.T, dot(linalg.inv(sigma), deviation))) / temperature
    if log_likelihood.imag != 0:
        print('warning: ll is complex! something is wrong.')
        log_likelihood = 1e200;
    
    return log_likelihood
