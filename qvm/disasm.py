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
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        bcode = f.read()

    module = QModule.parse(bcode)
    result = module.disassemble()
    print(result)


if __name__ == '__main__':
    main()
