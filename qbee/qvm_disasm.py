import argparse
from .qvm_module import QModule
from .utils import eprint


def perror(msg):
    eprint(msg)
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
