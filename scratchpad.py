#!/usr/bin/python

from numpy import *
import covariances, scenario, likelihoods, observation


stations = array(((0,0), (0,1), (1,0), (1,1)))
cov = covariances.genSECovar(1,.2)

euclidean_distance = lambda x,y: linalg.norm(x-y)
sc = scenario.Scenario(stations, cov, euclidean_distance, 0)

events = matrix( (.5, .5, 0) )
obs = observation.Observation( (.71, .71, .71, .71) )

ll = likelihoods.gp_likelihood(obs, events, sc)
print ll
