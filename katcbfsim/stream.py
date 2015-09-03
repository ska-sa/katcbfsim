"""Output stream abstraction"""

import trollius

class FXStreamSpeadFactory(object):
    """Data stream from an FX correlator, sent over SPEAD."""
    def __init__(self, endpoints):
        self._endpoints = endpoints

    @property
    def endpoints(self):
        return self._endpoints

    def __call__(self, *args, **kwargs):
        return FXStreamSpead(self._endpoints, *args, **kwargs)


class FXStreamSpead(object):
    def __init__(self, endpoints, n_antennas, n_baselines, accumulation_length):
        pass

    @trollius.coroutine
    def send(self, vis):
        if False:
            yield None   # Forces this to be a coroutine
        # TODO: fill in
