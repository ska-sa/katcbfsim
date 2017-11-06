#!/usr/bin/env python3

import numpy as np
import katconf
import katpoint
import logging
import os
from katcbfsim import rime
from katsdpsigproc import accel


def get_antennas_katconfig_meerkat():
    """Create MeerKAT antennas from katconfig files. This is currently a bit
    nasty (doesn't use the array config) because I couldn't make katconf's
    ArrayConfig work.
    """
    def make_antenna(num):
        path = os.path.join('static', 'antennas', 'm{:03}.conf'.format(num))
        conf = katconf.environ().resource_config(path)
        diam_str = conf.get('antenna', 'dish_diameter')
        diam = float(diam_str)
        pos_str = conf.get('antenna', 'nominal_position')
        lat, lng, alt = [float(x) for x in pos_str.split(',')]
        lat = katpoint.deg2rad(lat)
        lng = katpoint.deg2rad(lng)
        return katpoint.Antenna('m{:03}'.format(num), lat, lng, alt, diam)

    return [make_antenna(i) for i in range(64)]

def main():
    logging.basicConfig(level=logging.DEBUG)
    context = accel.create_some_context()
    queue = context.create_command_queue()
    template = rime.RimeTemplate(context, 64)
    antennas = get_antennas_katconfig_meerkat()
    sources = [katpoint.construct_radec_target(katpoint.deg2rad(18 + i), katpoint.deg2rad(-34))
               for i in range(64)]
    fn = template.instantiate(queue, 1284000000, 856000000, 32768, sources, antennas)
    fn.ensure_all_bound()
    fn.set_phase_center(sources[0])
    for i in range(10):
        fn()
        fn.set_time(fn.time + 0.5)

if __name__ == '__main__':
    main()
