import argparse
import sys
from .module import QModule


def perror(msg):
    print(msg, file=sys.stderr)
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Disassemble QVM code')

    parser.add_argument('input', help='File to read code from')
    parser.add_argument(
        '--headers', '-H', action='store_true',
        help='Only print header info')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        bcode = f.read()

    module = QModule.parse(bcode)

    if args.headers:
        print('literals:', len(module.literals))
        print('n_global_cells:', module.n_global_cells)
        data_parts = len(module.data)
        data_items = sum(len(i) for i in module.data)
        print('data parts:', data_parts, '  data items:', data_items)
    else:
        result = module.disassemble()
        print(result)


if __name__ == '__main__':
    main()
