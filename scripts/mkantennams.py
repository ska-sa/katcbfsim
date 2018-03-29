#!/usr/bin/env python3

import katpoint
import casacore.tables as tables
import numpy as np


def get_antennas_meerkat():
    """Create MeerKAT antennas from katconfig files. This is currently a bit
    nasty (doesn't use the array config) because I couldn't make katconf's
    ArrayConfig work.
    """
    with open('antennas.txt') as f:
        return [katpoint.Antenna(x) for x in f]


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
    return tables.maketabdesc([name, station, type_, mount, position, offset,
                               dish_diameter, flag_row])


def main():
    antennas = get_antennas_meerkat()
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
