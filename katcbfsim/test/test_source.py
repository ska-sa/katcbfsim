"""Tests for :mod:`katpoint.source`."""

from nose.tools import *
import numpy as np
import katpoint
from katcbfsim.source import Source


class TestSource(object):
    def setup(self):
        self.simple_source = Source('radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1.0 1.0)')
        self.pol_source = Source('{"Q": 0.3, "U": 0.2, "V": -0.1} radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1.0 1.0)')

    def test_construct_from_target(self):
        """Must be possible to construct a Source from a Target"""
        target = katpoint.Target('radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1.0 1.0)')
        source = Source(target)
        assert_equal(source, target)
        assert_equal([1.0, 0.0, 0.0, 0.0], source.stokes_scale)

    def test_construct_bad_json(self):
        """Must raise :exc:`ValueError` if the JSON is malformed"""
        with assert_raises(ValueError):
            Source('{"Q": 0.3, "U": 0.2, "V" -0.1} radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1.0 1.0)')

    def test_construct_bad_schema(self):
        """Must raise :exc:`ValueError` if the JSON does not match the schema"""
        with assert_raises(ValueError):
            Source('{"Q": 0.3, "U": 0.2, "V": [-0.1]} radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1.0 1.0)')

    def test_description(self):
        assert_equal('radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1 1)',
                     self.simple_source.description)
        assert_equal('{"Q": 0.3, "U": 0.2, "V": -0.1} radec, 0:02:00.00, -30:00:00.0, (100.0 2000.0 1 1)',
                     self.pol_source.description)

    def test_flux_density_scalar(self):
        # simple source
        flux = self.simple_source.flux_density_stokes(10.0)
        np.testing.assert_allclose([np.nan, np.nan, np.nan, np.nan], flux)
        flux = self.simple_source.flux_density_stokes(1000.0)
        np.testing.assert_allclose([1e4, 0.0, 0.0, 0.0], flux)
        # polarized source
        flux = self.pol_source.flux_density_stokes(10.0)
        np.testing.assert_allclose([np.nan, np.nan, np.nan, np.nan], flux)
        flux = self.pol_source.flux_density_stokes(1000.0)
        np.testing.assert_allclose([1e4, 3e3, 2e3, -1e3], flux)

    def test_flux_density_vector(self):
        flux = self.pol_source.flux_density_stokes([10.0, 1000.0, 10000.0])
        np.testing.assert_allclose(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [1e4, 3e3, 2e3, -1e3],
                [np.nan, np.nan, np.nan, np.nan]
            ], flux)
