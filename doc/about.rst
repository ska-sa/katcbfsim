About katcbfsim
===============
katcbfsim is a correlator and beamformer simulator, which is intended to match
the output formats used by the MeerKAT CBF.

Features
--------

Correlator
^^^^^^^^^^
The correlator simulation currently models the following effects:

- Unpolarised point sources

- Statistical noise

- A simple antenna-independent Airy beam model (all antennas are assumed to
  point in the same direction, and thus share a perceived sky).

The following are currently implemented internally, but not yet exposed:

- A per-frequency per-antenna full-Jones gain

- System noise, via a single system equivalent flux density

- Polarised sources

The following are not (yet) modelled:

- Direction-dependent effects other than a primary beam

- Atmospheric effects

- Relativistic aberration

- Relativistic Doppler effect

- Time and frequency smearing (values are point-sampled)

- Extended sources

- Gain drift

- Pointing errors

- Frequency- or antenna-dependent system noise

- RFI

- Quantisation effects

In addition, a simplified statistical model of noise is used, which does not
account for correlations between visibilities.

Beamformer
^^^^^^^^^^
The beamformer simulation is much more basic. It simulates only the metadata,
and the data is meaningless. As such, the antennas, sources, target etc do not
need to be specified.

Installation
------------
Installation uses the normal setup.py, and so can be installed with pip or
other Python packaging tools.

A CUDA-capable GPU and corresponding drivers must be present. This is currently
the case for the beamformer simulation as well, even though it is not used in
this case.

Using katcbfsim
---------------
There are several ways to run a simulation, which can also be mixed.

1. Command-line only. In this mode, the target (phase centre) is fixed for the
   simulation, and it is also the pointing direction.

2. Using katcp_. The sources, antennas, target etc are specified through katcp
   commands. In this mode, it is possible to simulate multiple streams, and
   to sent visibilities to HDF5 files instead of the network.

3. Using katsdptelstate_. Static configuration is stored in the :attr:`config`
   dictionary, as if for command-line arguments. Time-varying configuration is
   supplied as timestamped sensors.

4. Mixed katcp_ and katsdptelstate_. In this case, static configuration can be
   loaded into katsdptelstate_ later on (as attributes, rather than in the
   :attr:`config` dictionary), and then latched by a katcp_ command.

5. Using katcbfsim as a library.

.. _katcp: https://katcp-python.readthedocs.io/en/v0.6.4/_downloads/bff0f2ba64c586770b519594f52b1ab7/NRF-KAT7-6.0-IFCE-002-Rev5.pdf
.. _katsdptelstate: https://github.com/ska-sa/katsdptelstate

The configuration is split into information describing the virtual world
(antennas, sources and so on) and information about the virtual correlator or
beamformer (called a "stream"). Because the katcp_ interface supports multiple
streams, the per-stream commands all take a stream name.

The world information needed is:

- The antennas. Each is given by a string that can be parsed by
  :class:`katpoint.Antenna <katpoint.antenna.Antenna>`.

- The point sources. Each is given by a string that can be parsed by
  :class:`katpoint.Target <katpoint.target.Target>`, and should include a flux
  model. The source is ignored for frequencies outside the support of the flux
  model. If no flux model is given, 1 Jy is assumed at all frequencies. If no
  sources are given, a point source is simulated at the initial phase centre.

- The sync time, as a UNIX timestamp. This is the start time of the first
  dump, and also the time reported as the `sync_time` in the SPEAD metadata.

- The target, i.e., phase centre. This is also a string that can be parsed by
  :class:`katpoint.Target <katpoint.target.Target>`. This can be changed over
  time.

- The pointing direction, used for the primary beam model. Like the target,
  this is a :class:`katpoint.Target <katpoint.target.Target>` string and can be
  changed over time. If it is never specified, it defaults to being the same as
  the target.

- A gain (scaling factor between flux densities and counts). Generated values
  in Jansky are converted to output values by scaling by this gain. It is
  expressed as a scale factor per Hz of channel bandwidth per second of
  integration time.

The stream information is:

- A name, which is used in katcp requests and sensor names.

- An ADC clock rate, bandwidth, number of channels, and centre frequency.

- A destination, which is a hostname and port for the SPEAD stream, or the
  name of an HDF5 file.

- For correlation:

  - An accumulation length for integrations, in seconds. The actual value is
    rounded in the same way that the MeerKAT correlator would.

- For beamforming:

  - The number of time samples included in each heap.
  - The number of bits per sample.

Command-line
^^^^^^^^^^^^
Run :program:`cbfsim.py` :option:`--help` to see the command-line options. Only
a few key options are documented here.

.. program:: cbfsim.py

.. option:: --create-fx-stream <NAME>

   This creates a correlator stream with the given name. If this option is not
   specified, then the katcp request :samp:`stream-create-correlator` must be
   used to create streams.

.. option:: --create-beamformer-stream <NAME>

   This is equivalent to :option:`--create-fx-stream` but for beamformer
   streams.

.. option:: --start

   Start the capture for the stream. If this option is not specified, the
   katcp request :samp:`capture-init` must be used to start the capture.

.. option:: --cbf-antenna <DESCRIPTION>

   Specify a single antenna. Repeat multiple times to specify multiple
   antennas.

.. option:: --cbf-antenna-file <FILENAME>

   Load antenna descriptions from a file that contains one per line.

.. option:: --cbf-antenna-mask <LIST>

   Comma-separated list of antenna names. They will be given a fake position,
   which can later be replaced using the
   :samp:`configure-subarray-from-telstate` katcp request (see below).

.. option:: --cbf-sim-source <DESCRIPTION>, --cbf-sim-source-file <FILENAME>

   These are similar, but for sources rather than antennas.

.. option:: --cbf-sim-gain <FACTOR>

   System-wide gain, as described above

Telescope state
^^^^^^^^^^^^^^^
Command-line options can be loaded through katsdptelstate_ in the standard
way. Antennas and sources are slightly different, however. The antennas must
be placed in a :attr:`cbf_antennas` key (in the :attr:`config` dictionary),
which is a list of dictionaries. Each dictionary has a :attr:`description`
key, which is the antenna string. This is to allow for future expansion. The
sources are similarly placed in a :attr:`cbf_sim_sources` key.

The target is read from the telescope state sensor :attr:`cbf_target`, using
the latest value strictly prior to the start of the dump. Thus, all values for
a simulation can be pre-loaded.

The pointing direction is specified by the telescope state sensors
:attr:`ant_pos_actual_scan_azim` and :attr:`ant_pos_actual_scan_elev`, where
`ant` is replaced by the name of the first antenna. These provide the azimuth
and elevation, in degrees, for the first antenna. In future, other antenna
directions might be used, but for now they are ignored.

katcp protocol
^^^^^^^^^^^^^^
Use the :samp:`?help` command to obtain a full list of commands. The general
flow is

1. Define a stream with :samp:`?stream-create-correlator`.

2. Set world and correlator static properties.

3. Start the data flow with :samp:`?capture-start`.

4. Set dynamic properties as the simulation proceeds.

5. Stop the data flow with :samp:`?capture-stop`.

Note that static properties cannot be changed while a capture is in progress,
but can be modified between captures.

Mixed katcp and telstate
^^^^^^^^^^^^^^^^^^^^^^^^
If the subarray static properties are not known at the time the simulator
process is started, they can still be loaded from telstate later, using the
:samp:`?configure-subarray-from-telstate` request. This takes an optional
parameter, which is a comma-separated list of antenna names, and requires that
:option:`--telstate` was given on the command line. If no antenna names are
listed, the names of the antennas configured at startup are used, replacing
their positions (this is a good match for :option:`--cbf-antenna-names`).

For each antenna named `name`, the attribute :samp:`{name}_observer` is used to
obtain the antenna. It can be specified as either a description string or an
antenna object.

This flow can also be used in the opposite direction: antennas configured on
the command line with :option:`--cbf-antenna` or :option:`--cbf-antenna-file`
are stored into telstate as :samp:`{name}_observer`.
