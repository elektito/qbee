import argparse
import re
from functools import wraps
from functools import reduce
from pyparsing import (
    ParserElement, CaselessKeyword, Literal, Regex, LineEnd, StringEnd,
    Word, Forward, FollowedBy, White, Group, Empty, Located, SkipTo,
    alphas, alphanums, delimited_list, lineno, col
)
from .exceptions import SyntaxError
from .expr import (
    Type, Operator, NumericLiteral, Identifier, BinaryOp, UnaryOp,
    StringLiteral,
)
from .stmt import BeepStmt, CallStmt, ClsStmt
from .program import Program, Label


# Enable memoization
ParserElement.enable_packrat()

# Space and tab constitute as whitespace (but not newline)
ParserElement.set_default_whitespace_chars(' \t')

# --- Grammar ---

# Keywords

and_kw = CaselessKeyword('and')
beep_kw = CaselessKeyword('beep')
call_kw = CaselessKeyword('call')
cls_kw = CaselessKeyword('cls')
data_kw = CaselessKeyword('data')
eqv_kw = CaselessKeyword('eqv')
imp_kw = CaselessKeyword('imp')
mod_kw = CaselessKeyword('mod')
not_kw = CaselessKeyword('not')
or_kw = CaselessKeyword('or')
rem_kw = CaselessKeyword('rem')
xor_kw = CaselessKeyword('xor')

# create a single rule matching all keywords
_kw_rules = [
    rule
    for name, rule in globals().items()
    if name.endswith('_kw')
]
keyword = reduce(lambda a, b: a | b, _kw_rules)

# Operators and punctuation

type_char = Regex(r'[%&$#!]')
single_quote = Literal("'")
comma = Literal(',')
colon = Literal(':')
lpar = Literal("(")
rpar = Literal(")")
plus = Literal('+')
minus = Literal('-')
mul = Literal('*')
div = Literal('/')
addsub_op = plus | minus
muldiv_op = mul | div
intdiv_op = Literal('\\')
exponent_op = Literal('^')
compare_op = Regex(r'(<=|>=|<>|><|=<|=>|<|>|=)')

numeric_literal = (
    Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?") +
    type_char[...]
)

string_literal = Regex(r'"[^"]*"')

identifier = ~keyword + Word(alphas, alphanums) + type_char[...]

# Arithmetic Expressions

expr = Forward()
expr_list = delimited_list(expr, delim=',')
func_call = identifier + lpar + Group(expr_list) + rpar
paren_expr = Group(
    lpar.suppress() + expr + rpar.suppress()
)
atom = (
    addsub_op[...] +
    (
        (func_call | numeric_literal | string_literal | identifier) |
        paren_expr
    )
)
exponent_expr = Forward()
exponent_expr <<= atom + (exponent_op + exponent_expr)[...]
unary_expr = addsub_op[...] + exponent_expr
muldiv_expr = unary_expr + (muldiv_op + unary_expr)[...]
intdiv_expr = muldiv_expr + (intdiv_op + muldiv_expr)[...]
mod_expr = intdiv_expr + (mod_kw + intdiv_expr)[...]
addsub_expr = mod_expr + (addsub_op + mod_expr)[...]
compare_expr = addsub_expr + (compare_op + addsub_expr)[...]
not_expr = not_kw[...] + compare_expr
and_expr = not_expr + (and_kw + not_expr)[...]
or_expr = and_expr + (and_kw + and_expr)[...]
xor_expr = or_expr + (xor_kw + or_expr)[...]
eqv_expr = xor_expr + (xor_kw + xor_expr)[...]
imp_expr = eqv_expr + (eqv_kw + eqv_expr)[...]
expr <<= Located(imp_expr)

# Statements

quoted_string = Regex(r'"[^"]+"')
unquoted_string = Regex(r'[^"\n:]+')
unclosed_quoted_string = Regex(r'"[^"\n]+') + FollowedBy(LineEnd())
data_clause = quoted_string | unquoted_string
data_stmt = data_kw + (data_clause | comma)[...] + unclosed_quoted_string[...]

rem_stmt = (rem_kw + SkipTo(LineEnd())).suppress()

beep_stmt = beep_kw

call_stmt = call_kw[...].suppress() + identifier + expr_list[...]

cls_stmt = cls_kw

stmt = Located(
    beep_stmt |
    call_stmt |
    cls_stmt |
    data_stmt |
    rem_stmt |
    Empty()
)
stmts = stmt + (colon + stmt)[...]

# Lines and program

comment = single_quote + SkipTo(LineEnd())

line_no = Located(Regex(r'\d+') + FollowedBy(White()))
label = Located(
    ~keyword +
    Regex(r'[a-z][a-z0-9]', re.I) +
    Literal(':').suppress()
)

line_rest = stmts
line_prefix = line_no | label
line_with_just_prefix = line_prefix
line_without_prefix = line_rest
line_with_prefix = line_prefix + line_rest
line_without_nl = (
    line_with_prefix |
    line_with_just_prefix |
    line_without_prefix
)
line = (
    line_without_nl +
    comment[...].suppress() +
    LineEnd()[1, ...].suppress()
)

program = line[...]

# --- Parse actions ---

def parse_action(rule):
    def wrapper(func):
        rule.add_parse_action(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return func
    return wrapper


@parse_action(expr)
def parse_expr(toks):
    loc_start, (tok,), loc_end = toks
    tok.loc_start = loc_start
    tok.loc_end = loc_end
    return tok


@parse_action(stmt)
def parse_stmt(toks):
    loc_start, toks, loc_end = toks
    if len(toks) == 0:
        return toks
    tok = toks[0]
    tok.loc_start = loc_start
    tok.loc_end = loc_end
    return tok


@parse_action(string_literal)
def parse_str_literal(s, loc, toks):
    return StringLiteral(toks[0][1:-1])


@parse_action(numeric_literal)
def parse_num_literal(s, loc, toks):
    type_char = None
    if len(toks) == 2:
        type_char = toks[1]
    if type_char == '$':
        raise SyntaxError(
            loc, 'Invalid type character for numeric literal')
    try:
        return NumericLiteral.parse(toks[0], type_char)
    except ValueError:
        # probably something like "2.1%"
        raise SyntaxError(loc, 'Illegal number')


@parse_action(identifier)
def parse_identifier(toks):
    if len(toks) == 1:
        name = toks[0]
        type = None
    else:
        name, type_char = toks
        type = Type.from_type_char(type_char)
    return Identifier(name, type)


@parse_action(addsub_expr)
@parse_action(muldiv_expr)
@parse_action(intdiv_expr)
@parse_action(mod_expr)
@parse_action(and_expr)
@parse_action(or_expr)
@parse_action(xor_expr)
@parse_action(eqv_expr)
@parse_action(imp_expr)
def parse_left_assoc_binary_expr(toks):
    assert len(toks) % 2 == 1
    node = toks[0]
    for i in range(1, len(toks), 2):
        op = Operator.binary_op_from_token(toks[i])
        node = BinaryOp(node, toks[i+1], op)
    return node


@parse_action(exponent_expr)
def parse_right_assoc_binary_expr(toks):
    assert len(toks) % 2 == 1
    node = toks[-1]
    for i in range(-2, -len(toks) - 1, -2):
        assert toks[i] == '^'
        node = BinaryOp(node, toks[i-1], Operator.EXP)
    return node


@parse_action(not_expr)
@parse_action(unary_expr)
def parse_unary_expr(toks):
    if len(toks) > 0 and toks[0] in ('-', '+', 'not'):
        ops = toks[:-1]
        node = toks[-1]
        for op in reversed(ops):
            node = UnaryOp(node, Operator.unary_op_from_token(op))
        return node


@parse_action(expr_list)
def parse_expr_list(toks):
    return list(toks)


@parse_action(stmts)
def parse_stmts(toks):
    ret = []
    for tok in toks:
        if tok != ':':
            ret.append(tok)
    return ret


@parse_action(beep_stmt)
def parse_call(toks):
    return BeepStmt()


@parse_action(call_stmt)
def parse_call(toks):
    return CallStmt(toks[0].name, toks[1:])


@parse_action(cls_stmt)
def parse_call(toks):
    return ClsStmt()


@parse_action(label)
def parse_label(toks):
    loc_start, (tok,), loc_end = toks
    label = Label(str(tok))
    label.loc_start = loc_start
    label.loc_end = loc_end
    return label


@parse_action(program)
def parse_program(self, toks):
    return Program(toks)


def main():
    parser = argparse.ArgumentParser(
        description='A script used for testing the grammar.')

    parser.add_argument('expr', help='The value to parse.')
    parser.add_argument(
        '--rule', '-r', default='program',
        help='The rule to use for parsing. Defaults to "%(default)s".')
    parser.add_argument(
        '--not-all', '-n', action='store_true',
        help='If set, the parse_all flag is set to false.')

    args = parser.parse_args()

    rule = globals().get(args.rule)
    if not isinstance(rule, ParserElement):
        print(f'No such rule found: {args.rule}')
        exit(1)

    result = rule.parse_string(args.expr, parse_all=not args.not_all)
    if hasattr(result, '__len__') and isinstance(result[0], Program):
        for node in result[0].nodes:
            print(node)
    else:
        print(result)


if __name__ == '__main__':
    main()
