#!/usr/bin/env python

from __future__ import print_function, division
import katconf
import katpoint
import casacore.tables as tables
import os
import numpy as np


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


def antenna_desc():
    name = tables.makescacoldesc('NAME', '')
    station = tables.makescacoldesc('STATION', '')
    type_ = tables.makescacoldesc('TYPE', '')
    mount = tables.makescacoldesc('MOUNT', '')
    position_keywords = {
            'QuantumUnits': ['m', 'm', 'm'],
            'MEASINFO': {'type': 'position', 'Ref': 'WGS84'}
        }
    position = tables.makearrcoldesc('POSITION', 0.0, shape=(3,), keywords=position_keywords)
    offset = tables.makearrcoldesc('OFFSET', 0.0, shape=(3,), keywords=position_keywords)
    dish_diameter = tables.makescacoldesc('DISH_DIAMETER', 0.0, keywords={'QuantumUnits': ['m']})
    flag_row = tables.makescacoldesc('FLAG_ROW', False)
    return tables.maketabdesc([name, station, type_, mount, position, offset, dish_diameter, flag_row])


def main():
    print("WARNING: these positions are probably not accurate (no delay model used)!""")
    antennas = get_antennas_katconfig_meerkat()
    with tables.table('ANTENNA', antenna_desc(), nrow=len(antennas)) as table:
        table.putcol('NAME', [x.name for x in antennas])
        table.putcol('STATION', ['MEERKAT' for x in antennas])
        table.putcol('TYPE', ['GROUND-BASED' for x in antennas])
        table.putcol('MOUNT', ['ALT-AZ' for x in antennas])
        table.putcol('POSITION', np.array([x.position_ecef for x in antennas]))
        table.putcol('OFFSET', np.array([[0, 0, 0] for x in antennas]))
        table.putcol('DISH_DIAMETER', [x.diameter for x in antennas])
        table.putcol('FLAG_ROW', [False for x in antennas])


if __name__ == '__main__':
    main()
