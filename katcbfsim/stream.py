"""Output stream abstraction"""

class FXStreamSpead(object):
    """Data stream from an FX correlator, sent over SPEAD."""
    def __init__(self, endpoints):
        self._endpoints = endpoints

    @property
    def endpoints(self):
        return self._endpoints

    # TODO: fill in the rest
