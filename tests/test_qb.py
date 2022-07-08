import re
from glob import glob
from collections import namedtuple
from itertools import product
from pytest import mark, raises, fixture
from qbee.exceptions import ErrorCode, SyntaxError, CompileError
from qbee.compiler import Compiler
from qvm.module import QModule
from qvm.cpu import HaltReason
from testvm import TestMachine
from qb_test_parser import parse_qb_test_file, TestCase

# Import codegen implementations to enable them (this is needed even
# though we don't directly use the imported module)
from qbee import qvm_codegen  # noqa


CASES_DIR = 'tests/test_cases/'

QTestCase = namedtuple('QTestCase', [
    'filename',
    'idx',
    'text',
    'expected_result',
    'expected_error'
])

all_cases = []

def get_fixture_id(f):
    optimization, debug_info = f
    wdinfo = 'withdbg' if debug_info else 'nodbg'
    return f'O{optimization}-{wdinfo}'

@fixture(params=product([0, 1, 2], [True, False]), ids=get_fixture_id)
def compiler(request):
    optimization_level, debug_info = request.param
    compiler = Compiler(
        codegen_name='qvm',
        optimization_level=optimization_level,
        debug_info=debug_info,
    )
    yield compiler


@fixture
def vm():
    test_machine = TestMachine()
    yield test_machine


def load_test_file(filename):
    t = parse_qb_test_file(filename)
    return t.cases


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


def get_test_id(case):
    if not isinstance(case, TestCase):
        return
    return f'{case.filename}::{case.idx}'


@mark.parametrize('case', get_cases('success'), ids=get_test_id)
def test_qb_success(compiler, vm, case):
    code = compiler.compile(case.source_code)
    bcode = bytes(code)
    module = QModule.parse(bcode)
    vm.init(module, case)
    if not case.no_run:
        vm.run()
        assert vm.cpu.halt_reason == HaltReason.INSTRUCTION

        if case.expected_io:
            assert vm.io == case.expected_io


@mark.parametrize('case', get_cases('trap'), ids=get_test_id)
def test_qb_trap(compiler, vm, case):
    code = compiler.compile(case.source_code)
    bcode = bytes(code)
    module = QModule.parse(bcode)
    vm.init(module, case)
    if not case.no_run:
        vm.run()
        assert vm.cpu.halt_reason == HaltReason.TRAP

        if case.expected_error is not None:
            assert vm.cpu.last_trap == case.expected_error


@mark.parametrize('case', get_cases('syntaxerror'), ids=get_test_id)
def test_qb_syntax_error(compiler, case):
    with raises(SyntaxError) as e:
        compiler.compile(case.source_code)


@mark.parametrize('case', get_cases('compileerror'), ids=get_test_id)
def test_qb_compile_error(compiler, case):
    with raises(CompileError) as exc_info:
        compiler.compile(case.source_code)
    if case.expected_error is not None:
        assert exc_info.value.code == case.expected_error
