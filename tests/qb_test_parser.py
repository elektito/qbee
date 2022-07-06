from collections import namedtuple
from dataclasses import dataclass, field
from pyparsing import (
    ParserElement, Regex, Word, Literal, SkipTo, StringEnd, Opt, Group,
    QuotedString, delimited_list, alphas, alphanums,
)
from qbee.exceptions import ErrorCode
from qvm.trap import TrapCode


# Enable memoization
ParserElement.enable_packrat()

# Space and tab constitute as whitespace (but not newline)
ParserElement.set_default_whitespace_chars(' \t')


# parse actions

CaseOption = namedtuple('CaseOption', ['name', 'value'])
FileHeader = namedtuple('FileHeader', ['option'])
SourceCode = namedtuple('SourceCode', ['code'])
EmptyTestCase = namedtuple('EmptyTestCase', [])


@dataclass
class TestCase:
    source_code: str
    expected_result: str
    expected_error: str = None
    expected_io: list = field(default_factory=list)
    no_run: bool = False
    filename: str = None
    idx: int = None


@dataclass
class TestFile:
    filename: str = None
    global_expected_result: str = None
    global_expected_error: str = None
    cases: list[TestCase] = None


def parse_number(toks):
    value, = toks
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_case_option(toks):
    name, value = toks
    return CaseOption(name, value)


def parse_file_header(toks):
    option, = toks
    return FileHeader(option)


def parse_source_code(toks):
    code, = toks
    return SourceCode(code.strip())


def parse_test_case(toks):
    source_code, *options = toks
    options = list(options)
    if options == [[]]:
        options = []

    if not source_code.code and not options:
        return EmptyTestCase()

    expected_result = None
    expected_error = None
    expected_io = []
    no_run = False
    for name, value in options:
        name = name.lower()
        if name == 'success':
            assert value is None, '"success" option cannot have a value'
            assert expected_result is None, \
                'Expected result already set'
            expected_result = 'success'
        elif name == 'syntaxerror':
            assert expected_result is None, \
                'Expected result already set'
            # ignore value
            expected_result = 'syntaxerror'
        elif name == 'compileerror':
            assert expected_result is None, \
                'Expected result already set'
            expected_result = 'compileerror'
            if value:
                assert len(value) == 1, \
                    'Error code should be a single item'
                value = value[0].strip().upper()
                expected_error = ErrorCode[value]
        elif name == 'trap':
            assert expected_result is None, \
                'Expected result already set'
            expected_result = 'trap'
            if value:
                value = value[0].strip().upper()
                expected_error = TrapCode[value]
        elif name == 'io':
            expected_io.append(tuple(value))
        elif name == 'norun':
            no_run = True
        else:
            assert False, f'Unknown option: {name}'

    return TestCase(
        source_code=source_code.code,
        expected_result=expected_result,
        expected_error=expected_error,
        expected_io=expected_io,
        no_run=no_run)


def parse_test_file(toks):
    header, *cases = toks

    expected_result = None
    expected_error = None
    if header:
        name, value = header.option
        name = name.lower()
        if name == 'success':
            assert value is None, '"success" option cannot have a value'
            assert expected_result is None, \
                'Expected result already set'
            expected_result = 'success'
        elif name == 'syntaxerror':
            assert expected_result is None, \
                'Expected result already set'
            # ignore value
            expected_result = 'syntaxerror'
        elif name == 'compileerror':
            assert expected_result is None, \
                'Expected result already set'
            expected_result = 'compileerror'
            if value:
                assert len(value) == 1, \
                    'Error code should be a single item'
                value = value[0].strip().upper()
                expected_error = ErrorCode[value]
        elif name == 'trap':
            assert expected_result is None, \
                'Expected result already set'
            expected_result = 'trap'
            if value:
                value = value[0].strip().upper()
                expected_error = TrapCode[value]

    new_cases = []
    for case in cases:
        if isinstance(case, EmptyTestCase):
            continue
        if case.expected_result is None:
            case.expected_result = expected_result
            case.expected_error = expected_error

        new_cases.append(case)

    return TestFile(global_expected_result=expected_result,
                    global_expected_error=expected_error,
                    cases=new_cases)


# rules

case_delim = Literal('\n===\n')
options_delim = Literal('\n---\n')
nl = Literal('\n')
colon = Literal(':')
comma = Literal(',')
sharp = Literal('#')

source_code = (
    SkipTo(options_delim | case_delim | StringEnd())
).set_name('source_code').set_parse_action(parse_source_code)

option_name = (
    Word(alphanums + '_')
).set_name('option_name')

identifier = (
    Word(alphas, alphanums + '_')
).set_name('identifier')

number = (
    Regex(r'[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?\d+)?')
).set_name('number').set_parse_action(parse_number)

quoted_string = (
    QuotedString(quote_char='"', esc_char='\\')
).set_name('quoted_string')

option_value = (
    identifier | number | quoted_string
).set_name('option_value')

case_option = (
    option_name +
    Opt(
        colon.suppress() -
        Group(delimited_list(option_value, delim=comma), aslist=True),
        default=None,
    )
).set_name('case_option').set_parse_action(parse_case_option)

test_case = (
    source_code +
    Opt(
        options_delim.suppress() +
        delimited_list(case_option, delim=nl),
        default=[]
    )
).set_name('test_case').set_parse_action(parse_test_case)

file_header = (
    sharp.suppress() + case_option + nl.suppress()
).set_name('file_header').set_parse_action(parse_file_header)

test_file = (
    Opt(file_header, default=None) +
    delimited_list(test_case, delim=case_delim) +
    nl[...].suppress()
).set_name('test_file').set_parse_action(parse_test_file)


def parse_qb_test_file(filename):
    with open(filename) as f:
        contents = f.read()

    parsed = test_file.parse_string(contents, parse_all=True)
    parsed = parsed[0]

    parsed.filename = filename
    for i, case in enumerate(parsed.cases):
        case.filename = filename
        case.idx = i

    return parsed
