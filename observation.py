from numpy import *

class Observation:
    """
    Stores observed quantities: picks and differences.
    """

    def __init__(self, picks):
        self.picks = array(picks)
