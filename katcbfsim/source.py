"""Augments :class:`katpoint.Target` with extra information e.g. polarization.
"""

from collections import OrderedDict
import json
import jsonschema
import katpoint
import numpy as np


SOURCE_OPTIONS_SCHEMA = {
    'type': 'object',
    'properties': {
        'Q': {'type': 'number'},
        'U': {'type': 'number'},
        'V': {'type': 'number'}
    }
}


class Source(katpoint.Target):
    """An astronomical source with position and flux information.

    This is based on :class:`katpoint.Target`, but contains extra data that
    is not (yet) represented in target objects. In particular, it contains
    information about polarization, but may later also include a full
    source model.

    A source is represented by an extended version of the target description
    string, in which extra information is encoded as a JSON dictionary and
    prepended to the target string. For example::

        {"Q": 0.3, "U": 0.2, "V": -0.1} radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1.0)

    indicates a source where Stokes Q, U and V are found by multiplying Stokes
    I by the corresponding scale factors. At present these are the only valid
    keys.

    Parameters
    ----------
    base : object
        This can be either a :class:`katpoint.Target`, or a description
        string as defined above.

    Raises
    ------
    ValueError
        if `base` could not be parsed
    """
    def __init__(self, base):
        self.stokes_scale = [1.0, 0.0, 0.0, 0.0]
        if isinstance(base, katpoint.Target):
            super(Source, self).__init__(base)
        elif base.startswith('{'):
            # raw_decode helps us detect the end of the JSON document
            decoder = json.JSONDecoder()
            value, split = decoder.raw_decode(base)
            super(Source, self).__init__(base[split:].lstrip())
            try:
                jsonschema.validate(value, SOURCE_OPTIONS_SCHEMA)
            except jsonschema.ValidationError as error:
                raise ValueError(str(error))
            self.stokes_scale[1] = value.get('Q', 0.0)
            self.stokes_scale[2] = value.get('U', 0.0)
            self.stokes_scale[3] = value.get('V', 0.0)
        else:
            super(Source, self).__init__(base)

    @property
    def description(self):
        options = OrderedDict()
        if any(self.stokes_scale[1:]):
            options['Q'] = self.stokes_scale[1]
            options['U'] = self.stokes_scale[2]
            options['V'] = self.stokes_scale[3]
        if options:
            descr = json.dumps(options) + ' '
        else:
            descr = ''
        descr += super(Source, self).description
        return descr

    def flux_density_stokes(self, freq_MHz):
        """Calculate flux density for given observation frequency.

        Parameters
        ----------
        freq_MHz : float, or sequence of floats
            Frequency at which to evaluate flux density, in MHz

        Returns
        -------
        flux_density : 1D or 2D array of floats
            Flux density in Jy, or np.nan if the frequency is out of range.
            The last dimension has size 4 and coordinates to the Stokes
            parameters I, Q, U, V.
        """
        flux_I = self.flux_density(freq_MHz)
        return np.multiply.outer(flux_I, self.stokes_scale)
