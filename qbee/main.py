import argparse
import sys
from .compiler import Compiler
from .exceptions import SyntaxError, CompileError
from .utils import eprint


def main():
    parser = argparse.ArgumentParser(
        description='Yet another QBASIC compiler',
    )

    parser.add_argument('input', help='The file to compile.')

    parser.add_argument(
        '--output', '-o', default=None,
        help='Output file. Default is either "a.mod" or "a.asm" '
        'depending  on the output type. If set to "-" will output to '
        'stdout.')

    parser.add_argument(
        '--optimize', '-O', type=int, default=0,
        help='Optimization level. Defaults to %(default)s.')

    parser.add_argument(
        '--asm', '-S', action='store_true',
        help='Output assembly text, instead of assembled code.')

    args = parser.parse_args()

    if args.output is None:
        if args.asm:
            args.output = 'a.asm'
        else:
            args.output = 'a.mod'

    with open(args.input, encoding='ascii') as f:
        input_string = f.read()

    compiler = Compiler(
        optimization_level=args.optimize,
    )

    try:
        code = compiler.compile(input_string)
    except (SyntaxError, CompileError) as e:
        error_type = {
            SyntaxError: 'Syntax',
            CompileError: 'Compile',
        }[type(e)]
        if e.loc_start and e.loc_end:
            eprint(f'{error_type} error from loc {e.loc_start} to loc '
                   f'{e.loc_end}: {e}')
        elif e.loc_start:
            eprint(f'{error_type} error at loc {e.loc_start} : {e}')
        else:
            eprint(f'{error_type} error: {e}')

        exit(1)

    if args.asm:
        output = str(code)
    else:
        output = bytes(code)

    if args.output == '-':
        if isinstance(output, str):
            sys.stdout.write(output)
        else:
            sys.stdout.buffer.write(output)
    else:
        flags = 'wb' if isinstance(output, bytes) else 'w'
        with open(args.output, flags) as f:
            f.write(output)


if __name__ == '__main__':
    main()
