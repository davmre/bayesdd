from numpy import *
from scipy.integrate import dblquad

class Scenario:
    """
    Represents the details of the model specific to a particular location or application domain.
    """
    
    def __init__(self, stations, slowness_covar, traveltime_model, pick_stddev):
        
        # list of station locations
        self.stations = array(stations)

        # function of two locations, giving covariance of the slowness GP
        self.slowness_covar = slowness_covar

        # function of two locations, giving traveltime prediction between them 
        self.traveltime_model = traveltime_model

        # standard deviation of model pick error
        self.pick_stddev = pick_stddev


    def calc_residual_covariance(self, s1, s2, pt1, pt2):
        """
        Calculate the covariance between two residuals, as induced by the slowness covariance function. This is an integral over the paths given by the two event/station pairs. 
        """
        d1 = linalg.norm(pt1 - self.stations[s1])
        d2 = linalg.norm(pt2 - self.stations[s2])
        af = lambda a,b: self.slowness_covar((1-a)*self.stations[s1] + a*pt1, (1-b)*self.stations[s2] + b*pt2)
        a = dblquad(af, 0, 1, lambda x: 0, lambda x: 1)
        return d1*d2*a[0]

    def calc_traveltime(self, pts1, pts2):
        ttimes = zeros( (len(pts1), len(pts2)) )
        
        for i,pt1 in enumerate(pts1):
            for j, pt2 in enumerate(pts2):
                ttimes[i,j] = self.traveltime_model(pt1, pt2)

        return ttimes

    def interpolate_residual_covariance
