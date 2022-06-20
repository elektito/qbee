import argparse
import logging
import logging.config
from .module import QModule
from .machine import QvmMachine


logger = logging.getLogger(__name__)


def config_logging(args):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            }
        },
        'handlers': {
            'default': {
                'level': args.log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            }
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': args.log_level,
                'propagate': True,
            },
        }
    })


def log_level_type(value):
    return value.upper()


def main():
    parser = argparse.ArgumentParser(
        description='Qbee Virtual Machine')

    parser.add_argument(
        'module_file', help='The qbee module file to run.')
    parser.add_argument(
        '--log-level', '-l', type=log_level_type, default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level. Defaults to %(default)s.')
    parser.add_argument(
        '--dumb', action='store_true',
        help='Use dumb terminal, instead of smart terminal.')

    args = parser.parse_args()

    config_logging(args)

    with open(args.module_file, 'rb') as f:
        bcode = f.read()
        module = QModule.parse(bcode)

    terminal = 'dumb' if args.dumb else 'smart'
    machine = QvmMachine(module, terminal=terminal)

    machine.run()


if __name__ == '__main__':
    main()
