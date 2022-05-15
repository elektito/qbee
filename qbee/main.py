import argparse
from .compiler import Compiler
from .exceptions import SyntaxError, CompileError


def main():
    parser = argparse.ArgumentParser(
        description='Yet another QBASIC compiler',
    )

    parser.add_argument('input', help='The file to compile.')

    parser.add_argument(
        '--output', '-o', default='a.mod',
        help='Output file. Defaults to %(default)s.')

    parser.add_argument(
        '--optimize', '-O', type=int, default=0,
        help='Optimization level. Defaults to %(default)s.')

    args = parser.parse_args()

    with open(args.input, encoding='ascii') as f:
        input_string = f.read()

    compiler = Compiler(
        optimization_level=args.optimize,
    )

    try:
        module = compiler.compile(input_string)
    except (SyntaxError, CompileError) as e:
        error_type = {
            SyntaxError: 'Syntax',
            CompileError: 'Compile',
        }[type(e)]
        if e.loc_start and e.loc_end:
            print(f'{error_type} error from loc {e.loc_start} to loc '
                  f'{e.loc_end}: {e}')
        elif e.loc_start:
            print(f'{error_type} error at loc {e.loc_start} : {e}')
        else:
            print(f'{error_type} error: {e}')

        exit(1)

    with open(args.output, 'wb') as f:
        f.write(bytes(module))


if __name__ == '__main__':
    main()
