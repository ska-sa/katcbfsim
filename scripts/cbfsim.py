#!/usr/bin/env python3

import asyncio
import signal
import argparse
import logging
import time

from katsdpsigproc import accel
import katsdptelstate
import katsdpservices
import katpoint

import katcbfsim.server
from katcbfsim.stream import Subarray
from katcbfsim.source import Source
from katcbfsim.transport import DEFAULT_MAX_PACKET_SIZE


def parse_antenna(value):
    return {'description': value}


def parse_source(value):
    return {'description': value}


class TelstateSubarray(Subarray):
    """Overrides :meth:`target_at` to pull the target from a telescope state
    sensor."""
    def __init__(self, telstate, *args, **kwargs):
        super(TelstateSubarray, self).__init__(*args, **kwargs)
        self._telstate = telstate

    def target_at(self, timestamp):
        try:
            target = self._telstate.get_range('cbf_target', None, timestamp.secs)
            if target:
                return katpoint.Target(target[-1][0])  # Last element, value part of tuple
        except KeyError:
            pass
        # Failed, so fall back to the base class
        return super(TelstateSubarray, self).target_at(timestamp)

    def position_at(self, timestamp):
        try:
            antenna_name = self.antennas[0].name
            azim = self._telstate.get_range(antenna_name + '_pos_actual_scan_azim', None, timestamp.secs)
            elev = self._telstate.get_range(antenna_name + '_pos_actual_scan_elev', None, timestamp.secs)
            if azim and elev:
                # [-1][0] gives last element, value part of tuple
                azim = katpoint.deg2rad(azim[-1][0])
                elev = katpoint.deg2rad(elev[-1][0])
                return katpoint.construct_azel_target(azim, elev)
        except KeyError:
            pass
        # Failed, so fall back to the base class
        return super(TelstateSubarray, self).position_at(timestamp)


def prepare_server(server, args):
    """Do server configuration specified by command-line configuration"""
    # Do the fake ones first, so that real values can replace them
    if args.antenna_mask is not None:
        for name in args.antenna_mask.split(','):
            descr = name + ', 0:00:00.0, 00:00:00.0, 0.0, 0.0, , , 1.0'
            server.add_antenna(katpoint.Antenna(descr))
    for antenna in args.cbf_antennas:
        server.add_antenna(katpoint.Antenna(antenna['description']))
    if args.cbf_antenna_file is not None:
        with open(args.cbf_antenna_file) as f:
            for line in f:
                server.add_antenna(katpoint.Antenna(line))
    for source in args.cbf_sim_sources:
        server.add_source(Source(source['description']))
    if args.cbf_sim_source_file is not None:
        with open(args.cbf_sim_source_file) as f:
            for line in f:
                server.add_source(Source(line))
    if args.cbf_sync_time is None:
        args.cbf_sync_time = time.time()
    server.set_sync_time(args.cbf_sync_time)
    if args.cbf_start_time is None:
        args.cbf_start_time = args.cbf_sync_time
    server.set_gain(args.cbf_sim_gain)
    if args.cbf_target is not None:
        server.set_target(katpoint.Target(args.cbf_target))
    ifaddr = katsdpservices.get_interface_address(args.cbf_interface)
    if args.create_fx_stream is not None:
        stream = server.add_fx_stream(
            args.create_fx_stream,
            args.cbf_adc_sample_rate, args.cbf_center_freq, args.cbf_bandwidth,
            args.cbf_channels, args.cbf_substreams)
        server.set_accumulation_length(stream, args.cbf_int_time)
        server.set_destination(stream, args.cbf_spead, ifaddr, args.cbf_ibv,
                               args.max_packet_size)
        if args.dumps:
            server.set_n_dumps(stream, args.dumps)
        if args.start:
            server.capture_start(stream, katpoint.Timestamp(args.cbf_start_time))
    if args.create_beamformer_stream is not None:
        stream = server.add_beamformer_stream(
            args.create_beamformer_stream,
            args.cbf_adc_sample_rate, args.cbf_center_freq, args.cbf_bandwidth,
            args.cbf_channels, args.cbf_substreams,
            args.beamformer_timesteps, args.beamformer_bits)
        server.set_destination(stream, args.cbf_spead, ifaddr, args.cbf_ibv,
                               args.max_packet_size)
        if args.dumps:
            server.set_n_dumps(stream, args.dumps)
        if args.start:
            server.capture_start(stream, katpoint.Timestamp(args.cbf_start_time))
    if args.telstate is not None and (args.create_beamformer_stream or args.create_fx_stream):
        # Existing code may still depend on this. It's only done in the script
        # rather than by the streams themselves, because if the katcp interface
        # is used to add streams it's unknown how many streams the caller is
        # adding and hence when the fake sensors are fully "ready".
        args.telstate.add('sdp_cam2telstate_status', 'ready', immutable=False)


def configure_logging(level):
    katsdpservices.setup_logging()
    if level is not None:
        logging.root.setLevel(level.upper())


async def run_server(server):
    loop = asyncio.get_event_loop()
    await server.start()
    loop.add_signal_handler(signal.SIGINT, server.halt)
    loop.add_signal_handler(signal.SIGTERM, server.halt)
    await server.join()


def main():
    parser = katsdpservices.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--create-fx-stream', type=str, metavar='NAME', help='Create a correlator stream without prompting from katcp')
    group.add_argument('--create-beamformer-stream', type=str, metavar='NAME', help='Create a beamformer stream without prompting from katcp')
    parser.add_argument('--start', action='store_true', help='Start the defined streams')
    parser.add_argument('--dumps', type=int, help='Set finite number of dumps to produce for pre-configured streams [infinite]')
    parser.add_argument('--cbf-channels', type=int, default=32768, metavar='N', help='Number of channels [%(default)s]')
    parser.add_argument('--cbf-adc-sample-rate', type=float, default=1712000000.0, metavar='HZ', help='ADC rate [%(default)s]'),
    parser.add_argument('--cbf-bandwidth', type=float, default=856000000.0, metavar='HZ', help='Bandwidth [%(default)s]')
    parser.add_argument('--cbf-center-freq', type=float, default=1284000000.0, metavar='HZ', help='Sky center frequency [%(default)s]')
    parser.add_argument('--cbf-spead', type=katsdptelstate.endpoint.endpoint_list_parser(7148), metavar='ENDPOINT', default='127.0.0.1:7148', help='destination for CBF output [%(default)s]')
    parser.add_argument('--cbf-interface', metavar='INTERFACE', help='Network interface on which to send data [auto]')
    parser.add_argument('--cbf-ibv', action='store_true', help='Use ibverbs for acceleration (requires --cbf-interface)')
    parser.add_argument('--cbf-sync-time', type=float, metavar='TIME', help='Sync time as UNIX timestamp [now]')
    parser.add_argument('--cbf-start-time', type=float, metavar='TIME', help='Start time of stream as UNIX timestamp [sync time]')
    parser.add_argument('--cbf-int-time', type=float, metavar='TIME', default=0.5, help='Integration time in seconds [%(default)s]')
    parser.add_argument('--cbf-substreams', type=int, metavar='N', help='Number of substreams (X/B-engines) in simulated CBF [auto]')
    parser.add_argument('--cbf-antenna', dest='cbf_antennas', type=parse_antenna, action='append', default=[], metavar='DESCRIPTION', help='Specify an antenna (can be used multiple times)')
    parser.add_argument('--cbf-antenna-file', metavar='FILE', help='Load antenna descriptions from file, one per line')
    parser.add_argument('--cbf-antenna-names', dest='antenna_mask', metavar='NAME,...', help='Comma-separated list of antenna names to fake')
    parser.add_argument('--cbf-sim-source', dest='cbf_sim_sources', type=parse_source, action='append', default=[], metavar='DESCRIPTION', help='Specify a source object (can be used multiple times)')
    parser.add_argument('--cbf-sim-source-file', metavar='FILE', help='Load source descriptions from file, one per line')
    parser.add_argument('--cbf-sim-clock-ratio', type=float, default=1.0, metavar='RATIO', help='Ratio of real time to simulated time (<1 to run faster than real time, >1 for slower)')
    parser.add_argument('--cbf-sim-gain', type=float, default=1e-2, metavar='GAIN', help='Expected visibility for integrating 1Hz for 1s with a 1Jy source')
    parser.add_argument('--cbf-target', metavar='DESCRIPTION', help='Set initial target')
    parser.add_argument('--beamformer-timesteps', metavar='TIMES', type=int, default=256, help='Spectra included in each beamformer heap [%(default)s]')
    parser.add_argument('--beamformer-bits', metavar='BITS', type=int, choices=[8, 16, 32], default=8, help='Bits per real value in beamformer data [%(default)s]')
    parser.add_argument('--max-packet-size', metavar='BYTES', type=int, default=DEFAULT_MAX_PACKET_SIZE, help='Maximum SPEAD packet size for streams defined on command line [%(default)s]')
    parser.add_argument('--servers', metavar='N', type=int, default=1, help='Number of servers over which the simulation is split [%(default)s]')
    parser.add_argument('--server-id', metavar='N', type=int, default=1, help='Index of this server amongst --servers (1-based) [%(default)s]')
    parser.add_argument('--port', '-p', type=int, default=7147, help='katcp host port [%(default)s]')
    parser.add_argument('--host', '-a', type=str, default='', help='katcp host address [all hosts]')
    parser.add_argument('--log-level', '-l', default='INFO', help='logging level [%(default)s]')
    args = parser.parse_args()
    if args.start and args.create_fx_stream is None and args.create_beamformer_stream is None:
        parser.error('--start requires --create-fx-stream or --create-beamformer-stream')
    if args.cbf_ibv and args.cbf_interface is None:
        parser.error('--cbf-ibv requires --cbf-interface')
    configure_logging(args.log_level)
    katsdpservices.setup_restart()

    try:
        context = accel.create_some_context(interactive=False)
    except:
        logging.warn('Could not create a device context. FX simulation will not be possible')
        context = None
    if args.telstate is not None:
        subarray = TelstateSubarray(args.telstate)
    else:
        subarray = Subarray()
    subarray.clock_ratio = args.cbf_sim_clock_ratio
    subarray.n_servers = args.servers
    subarray.server_id = args.server_id - 1
    server = katcbfsim.server.SimulatorServer(
        context, subarray, telstate=args.telstate, host=args.host, port=args.port)
    prepare_server(server, args)
    asyncio.get_event_loop().run_until_complete(run_server(server))
    asyncio.get_event_loop().close()


if __name__ == '__main__':
    main()
