About katcbfsim
===============
katcbfsim is a correlator simulator, which is intended to match the output
formats used by the MeerKAT CBF. It currently models the following effects:

- Unpolarised point sources

- Statistical noise

The following are currently implemented internally, but not yet exposed:

- A per-frequency per-antenna full-Jones gain

- System noise, via a single system equivalent flux density

- Polarised sources

The following are not (yet) modelled:

- Direction-dependent effects, such as a beam model

- Atmospheric effects

- Relativistic aberration

- Relativistic Doppler effect

- Time and frequency smearing (values are point-sampled)

- Separation between simulation rate and 

- Extended sources

- Gain drift

- Pointing errors

- Frequency- or antenna-dependent system noise

- RFI

Installation
------------
Installation uses the normal setup.py, with the caveat that katcp must be at
least 0.6.x (which is not yet released on PyPI, and hence cannot be expressed
as a setuptools requirement).

Using katcbfsim
---------------
There are several ways to run a simulation:

1. Command-line only. In this mode, the target (phase centre) is fixed for the
   simulation.

2. Using katcp_. The sources, antennas, target etc are specified through katcp
   commands. In this mode, it is possible to simulate multiple products, and
   to sent visibilities to HDF5 files instead of the network.

3. Using katsdptelstate_. Static configuration is stored in the :attr:`config`
   dictionary, as if for command-line arguments. Time-varying configuration is
   supplied as timestamped sensors.

4. Using katcbfsim as a library.

.. _katcp: https://pythonhosted.org/katcp/
.. _katsdptelstate: https://github.com/ska-sa/katsdptelstate

The first three methods can also be freely mixed.

The configuration is split into information describing the virtual world
(antennas, sources and so on) and information about the virtual correlator
(called a "product"). Because the katcp_ interface supports multiple products,
the per-product commands all take a product name.

The world information needed is:

- The antennas. Each is given by a string that can be parsed by
  :class:`katpoint.Antenna`.

- The point sources. Each is given by a string that can be parsed by
  :class:`katpoint.Target`, and should include a flux model. The source is
  ignored for frequencies outside the support of the flux model. If no flux
  model is given, 1 Jy is assumed at all frequencies.

- The sync time, as a UNIX timestamp. This is the start time of the first
  dump, and also the time reported as the `sync_time` in the SPEAD metadata.

- The target, i.e., phase centre. In future this will also be the centre of a
  simulated beam. This is also a string that can be parsed by
  :class:`katpoint.Target`. This can be changed over time.

The product information is:

- A name, which is used in katcp requests and sensor names.

- An ADC clock rate, bandwidth, number of channels, and centre frequency.

- A destination, which is a hostname and port for the SPEAD stream, or the
  name of an HDF5 file.

- An accumulation length for integrations, in seconds. The actual value is
  rounded in the same way that the MeerKAT correlator would.

Command-line
^^^^^^^^^^^^
Run :program:`cbfsim.py` :option:`--help` to see the names of the
command-line options. Only a few options will be documented here.

.. program:: cbfsim.py

.. option:: --create-fx-product <NAME>

   This creates a product with the given name. If this option is not specified,
   then the katcp request :samp:`product-create-correlator` must be used to
   create products.

.. option:: --start

   Start the capture for the product. If this option is not specified, the
   katcp request :samp:`capture-init` must be used to start the capture.

.. option:: --cbf-antenna <DESCRIPTION>

   Specify a single antenna. Repeat multiple times to specify multiple
   antennas.

.. option:: --cbf-antenna-file <FILENAME>

   Load antenna descriptions from a file that contains one per line.

.. option:: --cbf-sim-source <DESCRIPTION>, --cbf-sim-source-file <FILENAME>

   These are similar, but for sources rather than antennas.

Telescope state
^^^^^^^^^^^^^^^
Command-line options can be loaded through katsdptelstate_ in the standard
way. Antennas and sources are slightly different, however. The antennas must
be placed in a `cbf_antennas` key, which is a list of dictionaries. Each
dictionary has a `description` key, which is the antenna string. This is to
allow for future expansion. The sources are similarly placed in a
`cbf_sim_sources` key.

The target is read from the telescope state sensor `cbf_target`, using the
latest value strictly prior to the start of the dump. Thus, all values for a
simulation can be pre-loaded.

katcp protocol
^^^^^^^^^^^^^^
Use the :samp:`?help` command to obtain a full list of commands. The general
flow is

1. Define a product with :samp:`?product-create-correlator`.

2. Set world and correlator static properties.

3. Start the data flow with :samp:`?capture-start`.

4. Set dynamic properties as the simulation proceeds.

5. Stop the data flow with :samp:`?capture-stop`.

Note that static properties cannot be changed while a capture is in progress,
but can be modified between captures.
