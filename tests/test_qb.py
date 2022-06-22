import re
from glob import glob
from collections import namedtuple
from itertools import product
from pytest import mark, raises, fixture
from qbee.exceptions import ErrorCode, SyntaxError, CompileError
from qbee.compiler import Compiler

# Import codegen implementations to enable them (this is needed even
# though we don't directly use the imported module)
from qbee import qvm_codegen  # noqa


CASES_DIR = 'tests/test_cases/'

QTestCase = namedtuple('QTestCase', ['text',
                                     'expected_result',
                                     'expected_error'])

all_cases = []


@fixture(params=product([0, 1, 2], [True, False]))
def compiler(request):
    optimization_level, debug_info = request.param
    compiler = Compiler(
        codegen_name='qvm',
        optimization_level=optimization_level,
        debug_info=debug_info,
    )
    yield compiler


def load_test_case(text, defaults=None):
    results_re = re.compile(
        r'(?P<expected_result>\w+)(:(?P<options>.*))?',)

    expected_result = 'success'
    expected_error = None
    if defaults:
        m = results_re.match(defaults)
        expected_result = m.group('expected_result')
        expected_error = m.group('options')
        if expected_error:
            expected_error = ErrorCode[expected_error.strip()]

    sep = '\n---\n'
    if sep in text:
        text, extra = text.split(sep)
        extra = extra.split('\n')

        m = results_re.match(extra[0])
        assert m

        expected_result = m.group('expected_result').lower()
        assert expected_result in ['syntaxerror',
                                   'compileerror',
                                   'success']

        expected_error = None
        if m.group('options'):
            expected_error = m.group('options').upper().strip()
            expected_error = ErrorCode[expected_error]

    return QTestCase(text, expected_result, expected_error)


def load_test_file(filename):
    with open(filename, encoding='cp437') as f:
        text = f.read()

    defaults = None
    if text.startswith('#'):
        eol = text.index('\n')
        defaults = text[1:eol].strip()
        text = text[eol+1:]

    parts = text.split('\n===\n')
    return [load_test_case(part, defaults) for part in parts]


def get_cases(expected_result):
    global all_cases

    if not all_cases:
        all_cases = sum(
            (load_test_file(f) for f in glob(f'{CASES_DIR}/**.test')),
            []
        )

    return [
        case
        for case in all_cases
        if case.expected_result == expected_result
    ]


@mark.parametrize('case', get_cases('success'))
def test_qb_success(compiler, case):
    compiler.compile(case.text)


@mark.parametrize('case', get_cases('syntaxerror'))
def test_qb_syntax_error(compiler, case):
    with raises(SyntaxError) as e:
        compiler.compile(case.text)


@mark.parametrize('case', get_cases('compileerror'))
def test_qb_compile_error(compiler, case):
    with raises(CompileError) as exc_info:
        compiler.compile(case.text)
    if case.expected_error:
        assert exc_info.value.code == case.expected_error
