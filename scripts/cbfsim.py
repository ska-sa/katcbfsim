#!/usr/bin/env python

from __future__ import print_function, division
import trollius
import tornado
import tornado.gen
from tornado.platform.asyncio import AsyncIOMainLoop
import signal
import argparse
import logging
import katcbfsim.server
from katsdpsigproc import accel


@tornado.gen.coroutine
def on_shutdown(server):
    print('Shutting down')
    yield server.stop()
    trollius.get_event_loop().stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=7147, help='katcp host port [%(default)s]')
    parser.add_argument('--host', '-a', type=str, default='', help='katcp host address [all hosts]')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    context = accel.create_some_context(interactive=False)
    ioloop = AsyncIOMainLoop()
    ioloop.install()
    server = katcbfsim.server.SimulatorServer(context, args.host, args.port)
    server.set_concurrency_options(thread_safe=False, handler_thread=False)
    server.set_ioloop(ioloop)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, server))
    ioloop.add_callback(server.start)
    trollius.get_event_loop().run_forever()

if __name__ == '__main__':
    main()
