"""Tests for :mod:`katcbfsim.rime`."""

from katsdpsigproc import accel
from katsdpsigproc.test.test_accel import device_test, force_autotune
from katcbfsim import rime

class TestRime(object):
    """Tests for :class:`katcbfsim.rime.Rime`."""

    @device_test
    @force_autotune
    def test_autotune(self, context, queue):
        """Test that autotuner runs successfully"""
        rime.RimeTemplate(context, 64)
