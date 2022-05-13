import re
from functools import wraps
from functools import reduce
from pyparsing import (
    ParserElement, CaselessKeyword, Literal, Regex, LineEnd, StringEnd,
    Word, Forward, FollowedBy, White, Group, alphas, alphanums,
    delimited_list,
)
from .exceptions import SyntaxError
from .expr import (
    Type, Operator, ExprNode, NumericLiteral, Identifier, BinaryOp,
    UnaryOp,
)
from .stmt import BeepStmt, CallStmt, ClsStmt


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
xor_kw = CaselessKeyword('xor')

# create a single rule matching all keywords
_kw_rules = [
    rule
    for name, rule in globals().items()
    if name.endswith('_kw')
]
keyword = reduce(lambda a, b: a | b, _kw_rules)

# Operators and punctuation

type_char = Regex(r'[%&$#]')
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
    Regex(r'\+?[0-9]+') + type_char[...] +
    type_char[...] +
    FollowedBy(Regex(r'[^a-z_]', re.I) |
                  LineEnd() |
                  StringEnd())
)

string_literal = Regex(r'"[^"]*"')

identifier = Word(alphas, alphanums) + type_char[...]

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
unary_expr = (
    addsub_op[1, ...] + exponent_expr |
    exponent_expr
)
muldiv_expr = unary_expr + (muldiv_op + unary_expr)[...]
intdiv_expr = muldiv_expr + (intdiv_op + muldiv_expr)[...]
mod_expr = intdiv_expr + (mod_kw + intdiv_expr)[...]
addsub_expr = mod_expr + (addsub_op + mod_expr)[...]
compare_expr = addsub_expr + (compare_op + addsub_expr)[...]
not_expr = (
    not_kw[1, ...] + compare_expr |
    compare_expr
)
and_expr = not_expr + (and_kw + not_expr)[...]
or_expr = and_expr + (and_kw + and_expr)[...]
xor_expr = or_expr + (xor_kw + or_expr)[...]
eqv_expr = xor_expr + (xor_kw + xor_expr)[...]
imp_expr = eqv_expr + (eqv_kw + eqv_expr)[...]
expr <<= imp_expr

# Statements and program

quoted_string = Regex(r'"[^"]+"')
unquoted_string = Regex(r'[^"\n:]+')
unclosed_quoted_string = Regex(r'"[^"\n]+') + FollowedBy(LineEnd())
data_clause = quoted_string | unquoted_string
data_stmt = data_kw + (data_clause | comma)[...] + unclosed_quoted_string[...]

line_no = Regex(r'\d+') + FollowedBy(White())
label = ~keyword + Regex(r'[a-z][a-z0-9_]+:\s*', re.I)

beep_stmt = beep_kw

call_stmt = call_kw[...].suppress() + identifier + expr_list[...]

cls_stmt = cls_kw

stmt = (
    beep_stmt |
    call_stmt |
    cls_stmt |
    data_stmt
)
stmts = stmt + (colon + stmt)[...]

line_rest = stmts
line_prefix = line_no | label
line_with_just_prefix = line_prefix
line_without_prefix = line_rest
line_with_prefix = line_prefix + line_rest
line_without_nl = line_with_prefix | line_without_prefix | line_with_just_prefix
line = line_without_nl + LineEnd()[1, ...].suppress()

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


@parse_action(numeric_literal)
def parse_num_literal(s, loc, toks):
    type_char = None
    if len(toks) == 2:
        type_char = toks[1]
    if type_char == '$':
        raise SyntaxError(
            loc, 'Invalid type character for numeric literal')
    return NumericLiteral.parse(toks[0], type_char)


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
    ops = toks[:-1]
    node = toks[-1]
    for op in reversed(ops):
        node = UnaryOp(node, Operator.unary_op_from_token(op))
    return node


@parse_action(expr_list)
def parse_expr_list(toks):
    return list(toks)


@parse_action(beep_stmt)
def parse_call(toks):
    return BeepStmt()


@parse_action(call_stmt)
def parse_call(toks):
    return CallStmt(toks[0].name, toks[1:])


@parse_action(cls_stmt)
def parse_call(toks):
    return ClsStmt()


class Compiler:
    def get_identifier_type(self, name):
        return Type.LONG


ExprNode.compiler = Compiler()


r = program.parse_string('foo x$+y$, 2+(3-4)', parse_all=True)
print(r)



instrs = []
for x in r:
    instrs += x.compile()
for instr in instrs:
    if instr == ('possibly_conv',):
        continue
    print(instr)
