import argparse
import re
from functools import reduce
from pyparsing import (
    ParserElement, CaselessKeyword, Literal, Regex, LineEnd, StringEnd,
    Word, Forward, FollowedBy, White, Group, Located, SkipTo, Combine,
    Opt, ParseException, ParseSyntaxException, alphas, alphanums,
    delimited_list,
)
from .exceptions import SyntaxError
from .expr import (
    Type, Operator, NumericLiteral, BinaryOp, UnaryOp, Lvalue,
    StringLiteral, ParenthesizedExpr, ArrayPass, BuiltinFuncCall,
)
from .stmt import (
    AssignmentStmt, BeepStmt, CallStmt, ClsStmt, ColorStmt, ConstStmt,
    DataStmt, DeclareStmt, DimStmt, GotoStmt, IfStmt, ElseClause,
    IfBeginStmt, ElseStmt, ElseIfStmt, EndIfStmt, InputStmt, PrintStmt,
    SubStmt, VarDeclClause, AnyVarDeclClause, ArrayDimRange,
    EndSubStmt, ExitSubStmt, TypeStmt, EndTypeStmt, FunctionStmt,
    EndFunctionStmt, ExitFunctionStmt, DoStmt, LoopStmt, EndStmt,
    ForStmt, NextStmt, ViewPrintStmt, SelectStmt, SimpleCaseClause,
    RangeCaseClause, CompareCaseClause, CaseStmt, CaseElseStmt,
    EndSelectStmt, PrintSep, WhileStmt, WendStmt, DefTypeStmt,
    RandomizeStmt, GosubStmt, ReturnStmt, DefSegStmt, PokeStmt,
    ReadStmt, RestoreStmt, LocateStmt, ScreenStmt, WidthStmt, PlayStmt,
    ExitDoStmt, ExitForStmt,
)
from .program import Label, LineNo, Line


# Enable memoization
ParserElement.enable_packrat()

# Space and tab constitute as whitespace (but not newline)
ParserElement.set_default_whitespace_chars(' \t')

# --- Grammar ---

# Keywords

abs_kw = CaselessKeyword('abs')
access_kw = CaselessKeyword('access')
and_kw = CaselessKeyword('and')
any_kw = CaselessKeyword('any')
as_kw = CaselessKeyword('as')
asc_kw = CaselessKeyword('asc')
atn_kw = CaselessKeyword('atn')
base_kw = CaselessKeyword('base')
beep_kw = CaselessKeyword('beep')
binary_kw = CaselessKeyword('binary')
bload_kw = CaselessKeyword('bload')
bsave_kw = CaselessKeyword('bsave')
case_kw = CaselessKeyword('case')
call_kw = CaselessKeyword('call')
cdbl_kw = CaselessKeyword('cdbl')
clng_kw = CaselessKeyword('clng')
chain_kw = CaselessKeyword('chain')
chdir_kw = CaselessKeyword('chdir')
chr_dollar_kw = CaselessKeyword('chr$')
cint_kw = CaselessKeyword('cint')
circle_kw = CaselessKeyword('circle')
clear_kw = CaselessKeyword('clear')
close_kw = CaselessKeyword('close')
cls_kw = CaselessKeyword('cls')
color_kw = CaselessKeyword('color')
com_kw = CaselessKeyword('com')
cos_kw = CaselessKeyword('cos')
common_kw = CaselessKeyword('common')
const_kw = CaselessKeyword('const')
csng_kw = CaselessKeyword('csng')
csrlin_kw = CaselessKeyword('csrlin')
cvd_kw = CaselessKeyword('cvd')
cvdmbf_kw = CaselessKeyword('cvdmbf')
cvi_kw = CaselessKeyword('cvi')
cvl_kw = CaselessKeyword('cvl')
cvs_kw = CaselessKeyword('cvs')
cvsmbf_kw = CaselessKeyword('cvsmbf')
data_kw = CaselessKeyword('data')
date_dollar_kw = CaselessKeyword('date$')
declare_kw = CaselessKeyword('declare')
def_kw = CaselessKeyword('def')
defdbl_kw = CaselessKeyword('defdbl')
defint_kw = CaselessKeyword('defint')
deflng_kw = CaselessKeyword('deflng')
defsng_kw = CaselessKeyword('defsng')
defstr_kw = CaselessKeyword('defstr')
dim_kw = CaselessKeyword('dim')
do_kw = CaselessKeyword('do')
double_kw = CaselessKeyword('double')
draw_kw = CaselessKeyword('draw')
else_kw = CaselessKeyword('else')
elseif_kw = CaselessKeyword('elseif')
end_kw = CaselessKeyword('end')
environ_dollar_kw = CaselessKeyword('environ$')
environ_kw = CaselessKeyword('environ')
eof_kw = CaselessKeyword('eof')
eqv_kw = CaselessKeyword('eqv')
erase_kw = CaselessKeyword('erase')
erdev_dollar_kw = CaselessKeyword('erdev$')
erdev_kw = CaselessKeyword('erdev')
erl_kw = CaselessKeyword('erl')
err_kw = CaselessKeyword('err')
error_kw = CaselessKeyword('error')
exit_kw = CaselessKeyword('exit')
exp_kw = CaselessKeyword('exp')
field_kw = CaselessKeyword('field')
fileattr_kw = CaselessKeyword('fileattr')
files_kw = CaselessKeyword('files')
fix_kw = CaselessKeyword('fix')
for_kw = CaselessKeyword('for')
fre_kw = CaselessKeyword('fre')
freefile_kw = CaselessKeyword('freefile')
function_kw = CaselessKeyword('function')
get_kw = CaselessKeyword('get')
gosub_kw = CaselessKeyword('gosub')
goto_kw = CaselessKeyword('goto')
hex_dollar_kw = CaselessKeyword('hex$')
if_kw = CaselessKeyword('if')
imp_kw = CaselessKeyword('imp')
inkey_dollar_kw = CaselessKeyword('inkey$')
inp_kw = CaselessKeyword('inp')
input_dollar_kw = CaselessKeyword('input$')
input_kw = CaselessKeyword('input')
instr_kw = CaselessKeyword('instr')
integer_kw = CaselessKeyword('integer')
int_kw = CaselessKeyword('int')
ioctl_kw = CaselessKeyword('ioctl')
ioctl_dollar_kw = CaselessKeyword('ioctl$')
is_kw = CaselessKeyword('is')
key_kw = CaselessKeyword('key')
kill_kw = CaselessKeyword('kill')
lbound_kw = CaselessKeyword('lbound')
lcase_dollar_kw = CaselessKeyword('lcase$')
left_dollar_kw = CaselessKeyword('left$')
len_kw = CaselessKeyword('len')
let_kw = CaselessKeyword('let')
line_kw = CaselessKeyword('line')
list_kw = CaselessKeyword('list')
loc_kw = CaselessKeyword('loc')
locate_kw = CaselessKeyword('locate')
lock_kw = CaselessKeyword('lock')
lof_kw = CaselessKeyword('lof')
log_kw = CaselessKeyword('log')
long_kw = CaselessKeyword('long')
loop_kw = CaselessKeyword('loop')
lpos_kw = CaselessKeyword('lpos')
lprint_kw = CaselessKeyword('lprint')
lset_kw = CaselessKeyword('lset')
ltrim_dollar_kw = CaselessKeyword('ltrim$')
mid_dollar_kw = CaselessKeyword('mid$')
mkd_dollar_kw = CaselessKeyword('mkd$')
mkdir_kw = CaselessKeyword('mkdir')
mkdmbf_dollar_kw = CaselessKeyword('mkdmbf$')
mki_dollar_kw = CaselessKeyword('mki$')
mkl_dollar_kw = CaselessKeyword('mkl$')
mks_dollar_kw = CaselessKeyword('mks$')
mksmbf_dollar_kw = CaselessKeyword('mksmbf$')
mod_kw = CaselessKeyword('mod')
name_kw = CaselessKeyword('name')
next_kw = CaselessKeyword('next')
not_kw = CaselessKeyword('not')
oct_dollar_kw = CaselessKeyword('oct$')
off_kw = CaselessKeyword('off')
on_kw = CaselessKeyword('on')
open_kw = CaselessKeyword('open')
option_kw = CaselessKeyword('option')
or_kw = CaselessKeyword('or')
out_kw = CaselessKeyword('out')
output_kw = CaselessKeyword('output')
paint_kw = CaselessKeyword('paint')
palette_kw = CaselessKeyword('palette')
pcopy_kw = CaselessKeyword('pcopy')
peek_kw = CaselessKeyword('peek')
pen_kw = CaselessKeyword('pen')
play_kw = CaselessKeyword('play')
pmap_kw = CaselessKeyword('pmap')
point_kw = CaselessKeyword('point')
poke_kw = CaselessKeyword('poke')
pos_kw = CaselessKeyword('pos')
preset_kw = CaselessKeyword('preset')
print_kw = CaselessKeyword('print')
pset_kw = CaselessKeyword('pset')
put_kw = CaselessKeyword('put')
random_kw = CaselessKeyword('random')
randomize_kw = CaselessKeyword('randomize')
read_kw = CaselessKeyword('read')
redim_kw = CaselessKeyword('redim')
rem_kw = CaselessKeyword('rem')
reset_kw = CaselessKeyword('reset')
restore_kw = CaselessKeyword('restore')
resume_kw = CaselessKeyword('resume')
return_kw = CaselessKeyword('return')
right_dollar_kw = CaselessKeyword('right$')
rmdir_kw = CaselessKeyword('rmdir')
rnd_kw = CaselessKeyword('rnd')
rset_kw = CaselessKeyword('rset')
rtrim_dollar_kw = CaselessKeyword('rtrim$')
run_kw = CaselessKeyword('run')
screen_kw = CaselessKeyword('screen')
seek_kw = CaselessKeyword('seek')
seg_kw = CaselessKeyword('seg')
select_kw = CaselessKeyword('select')
sgn_kw = CaselessKeyword('sgn')
shared_kw = CaselessKeyword('shared')
shell_kw = CaselessKeyword('shell')
single_kw = CaselessKeyword('single')
sleep_kw = CaselessKeyword('sleep')
sound_kw = CaselessKeyword('sound')
space_dollar_kw = CaselessKeyword('space$')
spc_kw = CaselessKeyword('spc')
sqr_kw = CaselessKeyword('sqr')
static_kw = CaselessKeyword('static')
step_kw = CaselessKeyword('step')
stick_kw = CaselessKeyword('stick')
stop_kw = CaselessKeyword('stop')
str_dollar_kw = CaselessKeyword('str$')
strig_kw = CaselessKeyword('strig')
string_dollar_kw = CaselessKeyword('string$')
string_kw = CaselessKeyword('string')
sub_kw = CaselessKeyword('sub')
swap_kw = CaselessKeyword('swap')
system_kw = CaselessKeyword('system')
tab_kw = CaselessKeyword('tab')
tan_kw = CaselessKeyword('tan')
then_kw = CaselessKeyword('then')
time_dollar_kw = CaselessKeyword('time$')
timer_kw = CaselessKeyword('timer')
to_kw = CaselessKeyword('to')
troff_kw = CaselessKeyword('troff')
tron_kw = CaselessKeyword('tron')
type_kw = CaselessKeyword('type')
ubound_kw = CaselessKeyword('ubound')
ucase_dollar_kw = CaselessKeyword('ucase$')
unlock_kw = CaselessKeyword('unlock')
until_kw = CaselessKeyword('until')
using_kw = CaselessKeyword('using')
val_kw = CaselessKeyword('val')
varptr_dollar_kw = CaselessKeyword('varptr$')
varptr_kw = CaselessKeyword('varptr')
varseg_kw = CaselessKeyword('varseg')
view_kw = CaselessKeyword('view')
wait_kw = CaselessKeyword('wait')
wend_kw = CaselessKeyword('wend')
while_kw = CaselessKeyword('while')
width_kw = CaselessKeyword('width')
window_kw = CaselessKeyword('window')
write_kw = CaselessKeyword('write')
xor_kw = CaselessKeyword('xor')

# create a single rule matching all keywords
_kw_rules = [
    rule
    for name, rule in globals().items()
    if name.endswith('_kw')
]
keyword = reduce(lambda a, b: a | b, _kw_rules).set_name('keyword')

# built-in types
type_name = (
    integer_kw |
    long_kw |
    single_kw |
    double_kw |
    string_kw
).set_name('type_name')

# Operators and punctuation

type_char = Regex(r'[%&$#!]')
letter = Regex('[a-z]', re.I)
dash = Literal('-')
single_quote = Literal("'")
eq = Literal('=')
comma = Literal(',')
colon = Literal(':')
semicolon = Literal(';')
dot = Literal('.')
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
    (
        Regex(
            r'[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eEdD][+-]?\d+)?') |
        Regex(r"&[Hh][0-9a-fA-F]+(%&)?") |
        Regex(r"&[Oo][0-7]+(%&)?")
    ) + Opt(type_char, default=None)
)

string_literal = Regex(r'"[^"]*"')

untyped_identifier = (
    ~keyword + Word(alphas, alphanums)
)
typed_identifier = Combine(untyped_identifier + type_char)
identifier = (
    typed_identifier |
    untyped_identifier
).set_name('identifier')

type_name = (
    integer_kw |
    long_kw |
    single_kw |
    double_kw |
    string_kw |
    untyped_identifier
).set_name('type_name')

# Expressions

array_pass = Located(
    identifier +
    lpar.suppress() +
    rpar.suppress()
).set_name('array_pass')

lvalue = Forward().set_name('lvalue')
expr = Forward().set_name('expr')

expr_list = delimited_list(expr, delim=',', min=1)
builtin_func = Located(
    (
        asc_kw |
        chr_dollar_kw |
        inkey_dollar_kw |
        int_kw |
        lcase_dollar_kw |
        left_dollar_kw |
        len_kw |
        mid_dollar_kw |
        peek_kw |
        right_dollar_kw |
        rnd_kw |
        space_dollar_kw |
        str_dollar_kw |
        timer_kw |
        ucase_dollar_kw |
        val_kw
    ) +
    Opt(
        lpar.suppress() +
        Group(expr_list, aslist=True) +
        rpar.suppress(),
        default=None
    )
).set_name('builtin_func')
paren_expr = Located(
    lpar.suppress() - expr + rpar.suppress()
)
atom = Located(
    addsub_op[...] +
    (
        (builtin_func | lvalue | numeric_literal | string_literal) |
        paren_expr
    )
)
exponent_expr = Forward()
exponent_expr <<= Located(atom + (exponent_op + exponent_expr)[...])
not_expr = Forward()
unary_expr = Located(
    addsub_op[1, ...] - exponent_expr |
    addsub_op[1, ...] - not_expr |  # allow combination of the two
                                    # unaries without parentheses
    exponent_expr
).set_name('unary_expr')
muldiv_expr = Located(unary_expr - (muldiv_op - unary_expr)[...])
intdiv_expr = Located(muldiv_expr - (intdiv_op - muldiv_expr)[...])
mod_expr = Located(intdiv_expr - (mod_kw - intdiv_expr)[...])
addsub_expr = Located(mod_expr - (addsub_op - mod_expr)[...])
compare_expr = Located(addsub_expr - (compare_op - addsub_expr)[...])
not_expr <<= Located(not_kw[...] - compare_expr)
and_expr = Located(not_expr - (and_kw - not_expr)[...])
or_expr = Located(and_expr - (or_kw - and_expr)[...])
xor_expr = Located(or_expr - (xor_kw - or_expr)[...])
eqv_expr = Located(xor_expr - (eqv_kw - xor_expr)[...])
imp_expr = Located(eqv_expr - (imp_kw - eqv_expr)[...])
expr <<= (
    array_pass |
    imp_expr
)

array_indices = Group(
    lpar.suppress() +
    expr_list +
    rpar.suppress()
)
dotted_vars = Group(
    dot.suppress() +
    delimited_list(untyped_identifier, delim=dot)
)
lvalue <<= Located(
    identifier +
    Opt(array_indices, default=None) +
    Opt(dotted_vars, default=None)
).set_name('lvalue')

# Statements

stmt_group = Forward()

line_no_value = Regex(r'\d+')
line_no = Located(
    line_no_value - FollowedBy(White() | LineEnd())
).set_name('line_number')
label = Located(
    ~keyword +
    untyped_identifier +
    Literal(':').suppress()
).set_name('label')

comment = (
    single_quote + SkipTo(LineEnd() | StringEnd())
).suppress().set_name('comment')

quoted_string = Regex(r'"[^"]*"')
unquoted_string = Regex(r'[^"\n:]+')
unclosed_quoted_string = Regex(r'"[^"\n]+') + FollowedBy(LineEnd())
data_clause = quoted_string | unquoted_string
data_stmt = (
    data_kw.suppress() -
    (data_clause | comma)[...] +
    unclosed_quoted_string[...]
).set_name('data_stmt')

rem_stmt = (rem_kw + SkipTo(LineEnd())).suppress().set_name('rem_stmt')

assignment_stmt = (
    let_kw[0, 1].suppress() +
    lvalue +
    eq.suppress() -
    expr
).set_name('assignment')

beep_stmt = beep_kw

select_stmt = (
    select_kw.suppress() -
    case_kw.suppress() -
    expr
).set_name('select_stmt')
case_clause = Located(
    Group(is_kw + compare_op + expr) |
    Group(expr + to_kw + expr) |
    Group(expr)
).set_name('case_clause')
case_stmt = (
    case_kw.suppress() +
    delimited_list(case_clause, delim=comma)
).set_name('case_stmt')
case_else_stmt = (case_kw + else_kw).set_name('case_else_stmt')
end_select_stmt = (end_kw + select_kw).set_name('end_select_stmt')

call_stmt = (
    (
        call_kw.suppress() -
        identifier -
        (
            lpar.suppress() -
            expr_list -
            rpar.suppress()
        )[0, 1]
    ) |
    (
        identifier +
        expr_list[0, 1]
    )
).set_name('call_stmt')

cls_stmt = cls_kw

color_stmt = (
    (color_kw.suppress() + expr + comma + expr + comma - expr) |
    (color_kw.suppress() + expr + comma + comma - expr) |
    (color_kw.suppress() + expr + comma + expr) |
    (color_kw.suppress() + comma + comma - expr) |
    (color_kw.suppress() + comma + expr) |
    (color_kw.suppress() - expr)
).set_name('color_stmt')

const_stmt = (
    const_kw.suppress() -
    identifier -
    eq.suppress() -
    expr
).set_name('const_stmt')

def_seg_stmt = (
    def_kw.suppress() +
    seg_kw.suppress() +
    Opt(eq.suppress() - expr, default=None)
).set_name('def_seg_stmt')

letter_range = Group(
    letter + Opt(dash.suppress() - letter, default=None)
).set_name('letter_range')
deftype_stmt = (
    (defint_kw | deflng_kw | defsng_kw | defdbl_kw | defstr_kw) -
    delimited_list(letter_range, delim=comma)
).set_name('deftype_stmt')

array_dim_range = Located(Group(
    expr +
    Opt(
        to_kw.suppress() -
        expr
    )
))
array_dims = Group(
    lpar.suppress() +
    delimited_list(array_dim_range, delim=',') +
    rpar.suppress()
).set_name('array_dims')
var_decl = Located(
    (
        untyped_identifier +
        Opt(array_dims, default=None) +
        as_kw -
        type_name
    ) |
    (
        identifier +
        Opt(array_dims, default=None)
    )
).set_name('var_decl')
dim_stmt = (
    Group(
        (dim_kw + shared_kw) |
        (dim_kw) |
        (static_kw),
        aslist=True
    ) +
    delimited_list(var_decl, delim=comma)
).set_name('dim_stmt')

do_stmt = (
    do_kw.suppress() +
    Opt((while_kw | until_kw) - expr)
).set_name('do_stmt')
loop_stmt = (
    loop_kw.suppress() +
    Opt((while_kw | until_kw) - expr)
).set_name('loop_stmt')
exit_do_stmt = (
    exit_kw.suppress() + do_kw.suppress()
).set_name('exit_do_stmt')

end_stmt = (
    end_kw | system_kw
).set_name('end_stmt')

for_stmt = (
    for_kw.suppress() -
    Group(Located(identifier)) -
    eq.suppress() -
    expr -
    to_kw.suppress() -
    expr +
    Opt(step_kw.suppress() - expr, default=None)
).set_name('for_stmt')
next_stmt = (
    next_kw.suppress() + Opt(Located(identifier), default=None)
).set_name('next_stmt')
exit_for_stmt = (
    exit_kw.suppress() + for_kw.suppress()
).set_name('exit_for_stmt')

gosub_stmt = (
    gosub_kw.suppress() -
    (untyped_identifier | line_no_value)
).set_name('gosub_stmt')
return_stmt = (
    return_kw.suppress() -
    Opt(untyped_identifier | line_no_value, default=None)
).set_name('return_stmt')

goto_stmt = goto_kw.suppress() - (untyped_identifier | line_no_value)

else_clause = Located(
    else_kw -
    stmt_group[0, 1]
).set_name('else_clause')
if_stmt = (
    if_kw.suppress() +
    expr +
    then_kw.suppress() +
    stmt_group +
    else_clause[0, 1]
).set_name('if_stmt')

if_block_begin = (
    if_kw.suppress() +
    expr +
    then_kw.suppress()
).set_name('if_block_begin')
elseif_stmt = (
    elseif_kw.suppress() -
    expr -
    then_kw.suppress() -
    stmt_group[0, 1]
).set_name('elseif_stmt')
else_stmt = (else_kw).set_name('else_stmt')
end_if_stmt = (end_kw + if_kw).set_name('end_if_stmt')

input_stmt = (
    input_kw.suppress() -
    semicolon[0, 1] +
    (
        string_literal[0, 1] +
        (semicolon | comma)
    )[0, 1] +
    delimited_list(lvalue, delim=',')
).set_name('input_stmt')

locate_stmt = (
    locate_kw.suppress() - expr - comma.suppress() - expr
).set_name('locate_stmt')

play_stmt = (
    play_kw.suppress() -
    expr
).set_name('play_stmt')

poke_stmt = (
    poke_kw.suppress() - expr - comma.suppress() - expr
).set_name('poke_stmt')

print_sep = Located(
    semicolon | comma
).set_name('print_sep')
print_stmt = (
    print_kw.suppress() +
    Opt(using_kw - expr - semicolon) +
    (expr[0, 1] + print_sep)[...]
    + expr[0, 1]
).set_name('print_stmt')

randomize_stmt = (
    randomize_kw.suppress() -
    expr
).set_name('randomize_stmt')

read_stmt = (
    read_kw.suppress() -
    delimited_list(lvalue, delim=comma)
).set_name('read_stmt')

restore_stmt = (
    restore_kw.suppress() +
    Opt(untyped_identifier | line_no_value, default=None)
).set_name('restore_stmt')

screen_stmt = (
    screen_kw.suppress() -
    delimited_list(expr, delim=comma, min=1, max=4)
).set_name('screen_stmt')

view_print_stmt = (
    view_kw.suppress() +
    print_kw.suppress() +
    Opt(
        Group(expr + to_kw.suppress() - expr),
        default=None
    )
).set_name('view_print_stmt')

while_stmt = (
    while_kw.suppress() -
    expr
).set_name('while_stmt')
wend_stmt = (
    wend_kw
).set_name('wend_stmt')

width_stmt = (
    width_kw.suppress() -
    (
        (expr + comma + expr) |
        (comma + expr) |
        (expr)
    )
).set_name('width_stmt')

type_field_decl = Located(
    untyped_identifier +
    as_kw -
    type_name
).set_name('type_field_decl')

type_stmt = (
    type_kw.suppress() -
    untyped_identifier
).set_name('type_stmt')
end_type_stmt = (end_kw + type_kw).set_name('end_type_stmt')

param_decl = Located(
    (
        untyped_identifier +
        Group(lpar - rpar) +
        Opt(as_kw - type_name)
    ) |
    (
        identifier + Group(lpar - rpar)
    ) |
    var_decl
)
param_list = delimited_list(param_decl, delim=comma)
sub_stmt = (
    sub_kw.suppress() -
    untyped_identifier +
    Opt(Group(
        lpar.suppress() +
        param_list +
        rpar.suppress(),
        aslist=True
    ), default=None) +
    Opt(static_kw, default=None)
).set_name('sub_stmt')
end_sub_stmt = end_kw + sub_kw
exit_sub_stmt = exit_kw + sub_kw

function_stmt = (
    function_kw.suppress() -
    identifier +
    Opt(Group(
        lpar.suppress() +
        param_list +
        rpar.suppress(),
        aslist=True
    ), default=None) +
    Opt(static_kw, default=None)
).set_name('function_stmt')
end_function_stmt = end_kw + function_kw
exit_function_stmt = exit_kw + function_kw

declare_var_decl = Located(
    (
        untyped_identifier +
        Opt(Group(lpar + rpar)) +
        as_kw +
        any_kw
    ) |
    param_decl
)
declare_param_list = delimited_list(declare_var_decl, delim=comma)
declare_stmt = (
    declare_kw.suppress() -
    (
        (sub_kw + untyped_identifier) |
        (function_kw + identifier)
    ) +
    (
        lpar.suppress() +
        declare_param_list[0, 1] +
        rpar.suppress()
    )[0, 1]
).set_name('declare_stmt')


# program

stmt = Located(
    type_stmt |
    type_field_decl |  # only valid in type blocks; checked by compiler
    end_type_stmt |

    assignment_stmt |
    beep_stmt |
    call_stmt |
    cls_stmt |
    color_stmt |
    const_stmt |
    data_stmt |
    def_seg_stmt |
    declare_stmt |
    deftype_stmt |
    dim_stmt |

    do_stmt |
    loop_stmt |
    exit_do_stmt |

    while_stmt |
    wend_stmt |

    for_stmt |
    next_stmt |
    exit_for_stmt |

    # the order of the following is important. in particular, if_stmt
    # must be before if_block_begin.
    if_stmt |
    if_block_begin |
    else_stmt |
    elseif_stmt |
    end_if_stmt |

    select_stmt |
    case_stmt |
    case_else_stmt |
    end_select_stmt |

    randomize_stmt |
    read_stmt |
    restore_stmt |
    screen_stmt |
    sub_stmt |
    end_sub_stmt |
    exit_sub_stmt |
    function_stmt |
    end_function_stmt |
    exit_function_stmt |
    gosub_stmt |
    return_stmt |
    goto_stmt |
    input_stmt |
    locate_stmt |
    play_stmt |
    poke_stmt |
    print_stmt |
    rem_stmt |
    view_print_stmt |
    width_stmt |

    # this needs to be after all other statements starting with the
    # end keyword
    end_stmt
).set_name('stmt')

line_prefix = label | line_no
stmt_group <<= (
    stmt +
    (colon[1, ...].suppress() + stmt)[...] +
    colon[...].suppress()
).set_name('stmt_group')
line = (
    White()[...].suppress() +
    line_prefix[0, 1] +
    stmt_group[0, 1] +
    colon[...].suppress() +
    comment[0, 1] +
    LineEnd()
).set_name('line')


# --- Parse actions ---


def parse_action(rule):
    def wrapper(func):
        rule.add_parse_action(func)
        return func
    return wrapper


@parse_action(typed_identifier)
@parse_action(untyped_identifier)
def parse_identifier(toks):
    return [toks[0].lower()]


@parse_action(atom)
def parse_atom(toks):
    loc_start, toks, loc_end = toks
    return toks


@parse_action(builtin_func)
def parse_builtin_func(toks):
    loc_start, (func, args), loc_end = toks
    args = args or []
    call = BuiltinFuncCall(func, args)
    call.loc_start = loc_start
    call.loc_end = loc_end
    return call


@parse_action(paren_expr)
def parse_paren_expr(toks):
    loc_start, toks, loc_end = toks
    assert len(toks) == 1
    expr = ParenthesizedExpr(toks[0])
    expr.loc_start = loc_start
    expr.loc_end = loc_end
    return expr


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
    literal, type_char = toks
    if type_char == '$':
        raise SyntaxError(
            loc, 'Invalid type character for numeric literal')

    try:
        num = NumericLiteral.parse(literal, type_char)
    except ValueError as e:
        desc = str(e) or 'Illegal number'
        raise SyntaxError(loc, desc)

    return num


@parse_action(addsub_expr)
@parse_action(muldiv_expr)
@parse_action(intdiv_expr)
@parse_action(mod_expr)
@parse_action(and_expr)
@parse_action(or_expr)
@parse_action(xor_expr)
@parse_action(eqv_expr)
@parse_action(imp_expr)
@parse_action(compare_expr)
def parse_left_assoc_binary_expr(toks):
    loc_start, toks, loc_end = toks
    assert len(toks) % 2 == 1
    node = toks[0]
    for i in range(1, len(toks), 2):
        op = Operator.binary_op_from_token(toks[i])
        node = BinaryOp(node, toks[i+1], op)
    node.loc_start = loc_start
    node.loc_end = loc_end
    return node


@parse_action(exponent_expr)
def parse_right_assoc_binary_expr(toks):
    loc_start, toks, loc_end = toks
    assert len(toks) % 2 == 1
    node = toks[-1]
    for i in range(-2, -len(toks) - 1, -2):
        assert toks[i] == '^'
        node = BinaryOp(toks[i-1], node, Operator.EXP)
        node.loc_start = loc_start
        node.loc_end = loc_end
    node.loc_start = loc_start
    node.loc_end = loc_end
    return node


@parse_action(not_expr)
@parse_action(unary_expr)
def parse_unary_expr(toks):
    loc_start, toks, loc_end = toks
    if len(toks) > 0 and toks[0] in ('-', '+', 'not'):
        ops = toks[:-1]
        node = toks[-1]
        for op in reversed(ops):
            node = UnaryOp(node, Operator.unary_op_from_token(op))
            node.loc_start = loc_start
            node.loc_end = loc_end
        node.loc_start = loc_start
        node.loc_end = loc_end
        return node
    return toks


@parse_action(expr_list)
def parse_expr_list(toks):
    return list(toks)


@parse_action(dotted_vars)
def parse_dotted_vars(toks):
    # This is a workaround for a bug in pyparsing. The parse action
    # for identifier and untyped_identifier is not called when inside
    # dotted_vars at the moment, so we perform this action here, even
    # though it should have been done in parse_identifier already
    toks = toks[0]
    toks = [t.lower() for t in toks]
    return [toks]


@parse_action(lvalue)
def parse_lvalue(toks):
    loc_start, toks, loc_end = toks
    base_var, array_indices, dotted_vars = toks
    array_indices = array_indices or []
    dotted_vars = dotted_vars or []
    lvalue = Lvalue(base_var, list(array_indices), list(dotted_vars))
    lvalue.loc_start = loc_start
    lvalue.loc_end = loc_end
    return lvalue


@parse_action(assignment_stmt)
def parse_assignment(toks):
    lvalue, expr = toks
    return AssignmentStmt(lvalue, expr)


@parse_action(beep_stmt)
def parse_beep(toks):
    return BeepStmt()


@parse_action(select_stmt)
def parse_select_stmt(toks):
    return SelectStmt(toks[0])


@parse_action(case_clause)
def parse_case_clause(toks):
    loc_start, (toks,), loc_end = toks
    if len(toks) == 1:
        clause = SimpleCaseClause(toks[0])
    elif toks[0] == 'is':
        _, op, value = toks
        clause = CompareCaseClause(op, value)
    elif toks[1] == 'to':
        from_value, _, to_value = toks
        clause = RangeCaseClause(from_value, to_value)
    else:
        assert False

    clause.loc_start = loc_start
    clause.loc_end = loc_end

    return clause


@parse_action(case_stmt)
def parse_case_stmt(toks):
    return CaseStmt(list(toks))


@parse_action(case_else_stmt)
def parse_case_else(toks):
    return CaseElseStmt()


@parse_action(end_select_stmt)
def parse_end_select(toks):
    return EndSelectStmt()


@parse_action(array_pass)
def array_pass(toks):
    loc_start, toks, loc_end = toks
    array_pass = ArrayPass(toks[0])
    array_pass.loc_start = loc_start
    array_pass.loc_end = loc_end
    return array_pass


@parse_action(call_stmt)
def parse_call(toks):
    return CallStmt(toks[0], toks[1:])


@parse_action(cls_stmt)
def parse_cls(toks):
    return ClsStmt()


@parse_action(color_stmt)
def parse_color(toks):
    colors = []
    last_tok = None
    for tok in toks:
        if tok == ',':
            if last_tok is None:
                # first argument left out
                colors.append(None)
            elif tok == last_tok == ',':
                # middle argument left out
                colors.append(None)
        elif tok != ',':
            colors.append(tok)
        last_tok = tok

    # add any left out arguments at the end
    colors += [None] * (3 - len(colors))

    return ColorStmt(*colors)


@parse_action(const_stmt)
def parse_const_stmt(toks):
    name, value = toks
    return ConstStmt(name, value)


@parse_action(def_seg_stmt)
def parse_def_seg_stmt(toks):
    segment, = toks
    return DefSegStmt(segment)


@parse_action(declare_stmt)
def parse_declare(toks):
    routine_type, name, *var_decls = list(toks)
    return DeclareStmt(routine_type, name, var_decls)


@parse_action(deftype_stmt)
def parse_deftype(toks):
    kw, *ranges = toks

    letters = set()
    for start, end in ranges:
        if end:
            letters.update(
                chr(c) for c in range(ord(start), ord(end) + 1))
        else:
            letters.add(start)

    def_type = {
        'defint': Type.INTEGER,
        'deflng': Type.LONG,
        'defsng': Type.SINGLE,
        'defdbl': Type.DOUBLE,
        'defstr': Type.STRING,
    }[kw]
    return DefTypeStmt(def_type, letters)


@parse_action(dim_stmt)
def parse_dim(toks):
    kind, *toks = toks
    if kind == ['dim', 'shared']:
        kind = 'dim_shared'
    elif kind == ['dim']:
        kind = 'dim'
    elif kind == ['static']:
        kind = 'static'
    else:
        assert False
    var_decls = list(toks)
    return DimStmt(var_decls, kind)


@parse_action(do_stmt)
def parse_do_stmt(toks):
    if toks:
        kind, cond = toks
        return DoStmt(kind, cond)
    else:
        return DoStmt('forever', None)


@parse_action(loop_stmt)
def parse_loop_stmt(toks):
    if toks:
        kind, cond = toks
        return LoopStmt(kind, cond)
    else:
        return LoopStmt('forever', None)


@parse_action(exit_do_stmt)
def parse_exit_do(toks):
    return ExitDoStmt()


@parse_action(for_stmt)
def parse_for_stmt(toks):
    var, from_expr, to_expr, step_expr = toks

    var_loc_start, (var,), var_loc_end = var
    var = Lvalue(var, [], [])
    var.loc_start = var_loc_start
    var.loc_end = var_loc_end
    return ForStmt(var, from_expr, to_expr, step_expr)


@parse_action(next_stmt)
def parse_next_stmt(toks):
    if toks[0] is None:
        return NextStmt(None)

    var_loc_start, (var,), var_loc_end = toks
    var = Lvalue(var, [], [])
    var.loc_start = var_loc_start
    var.loc_end = var_loc_end

    return NextStmt(var)


@parse_action(exit_for_stmt)
def parse_exit_for(toks):
    return ExitForStmt()


@parse_action(end_stmt)
def parse_end_stmt(toks):
    return EndStmt()


@parse_action(gosub_stmt)
def parse_gosub(toks):
    target = toks[0]
    if target[0].isnumeric():
        # it's a line number
        target = int(target)
    return GosubStmt(target)


@parse_action(return_stmt)
def parse_return(toks):
    target = toks[0]
    if target and target[0].isnumeric():
        # it's a line number
        target = int(target)
    return ReturnStmt(target)


@parse_action(goto_stmt)
def parse_goto(toks):
    target = toks[0]
    if target[0].isnumeric():
        # it's a line number
        target = int(target)
    return GotoStmt(target)


@parse_action(if_stmt)
def parse_if(toks):
    toks = list(toks)
    cond, *stmts = toks
    if stmts and isinstance(stmts[-1], ElseClause):
        else_stmt = stmts[-1]
        stmts = stmts[:-1]
    else:
        else_stmt = None
    return IfStmt(cond, stmts, else_stmt)


@parse_action(if_block_begin)
def parse_if_block_begin(toks):
    cond = toks[0]
    return IfBeginStmt(cond)


@parse_action(else_stmt)
def parse_else_stmt(toks):
    return ElseStmt()


@parse_action(else_clause)
def parse_else_clause(toks):
    loc_start, toks, loc_end = toks
    stmts = list(toks[1:])  # drop else keyword
    else_clause = ElseClause(stmts)
    else_clause.loc_start = loc_start
    else_clause.loc_end = loc_end
    return else_clause


@parse_action(elseif_stmt)
def parse_elseif(toks):
    toks = list(toks)
    cond, *stmts = toks
    return ElseIfStmt(cond, stmts)


@parse_action(end_if_stmt)
def parse_end_if(toks):
    return EndIfStmt()


@parse_action(input_stmt)
def parse_input(toks):
    same_line = False
    if toks[0] == ';':
        same_line = True
        toks.pop(0)

    prompt = StringLiteral('')
    prompt_question = True
    if isinstance(toks[0], StringLiteral):
        prompt = toks.pop(0)
        sep = toks.pop(0)
        prompt_question = (sep == ';')

    var_list = list(toks)

    return InputStmt(same_line, prompt, prompt_question, var_list)


@parse_action(locate_stmt)
def parse_locate_stmt(toks):
    row, col = toks
    return LocateStmt(row, col)


@parse_action(play_stmt)
def parse_play_stmt(toks):
    command_string = toks[0]
    return PlayStmt(command_string)


@parse_action(poke_stmt)
def parse_poke_stmt(toks):
    address, value = toks
    return PokeStmt(address, value)


@parse_action(print_sep)
def parse_print_sep(toks):
    loc_start, (tok,), loc_end = toks
    sep = PrintSep(tok)
    sep.loc_start = loc_start
    sep.loc_end = loc_end
    return sep


@parse_action(print_stmt)
def parse_print(toks):
    items = list(toks)

    format_string = None
    if items and items[0] == 'using':
        using_kw, format_string, semicolon, *items = items

    return PrintStmt(items, format_string)


@parse_action(randomize_stmt)
def parse_randomize(toks):
    seed = toks[0]
    return RandomizeStmt(seed)


@parse_action(read_stmt)
def parse_read_stmt(toks):
    targets = toks
    return ReadStmt(list(targets))


@parse_action(restore_stmt)
def parse_restore_stmt(toks):
    target = toks[0]
    if target and target[0].isnumeric():
        # it's a line number
        target = int(target)
    return RestoreStmt(target)


@parse_action(screen_stmt)
def parse_screen_stmt(toks):
    if len(toks) == 1:
        mode = toks[0]
        color_switch = apage = vpage = None
    elif len(toks) == 2:
        mode, color_switch = toks
        apage = vpage = None
    elif len(toks) == 3:
        mode, color_switch, apage = toks
        vpage = None
    else:
        mode, color_switch, apage, vpage = toks

    return ScreenStmt(mode, color_switch, apage, vpage)


@parse_action(data_stmt)
def parse_data(s, loc, toks):
    # Re-join data items and have them properly parsed again
    s = ' '.join(str(t) for t in toks)
    ret = DataStmt(s)

    if ret.items is None:
        raise SyntaxError(
            loc,
            'Syntax Error in DATA. NOTE: In QBASIC this error is '
            'raised when executing the READ statement.')

    return ret


@parse_action(label)
def parse_label(toks):
    loc_start, (tok,), loc_end = toks
    label = Label(str(tok))
    label.loc_start = loc_start
    label.loc_end = loc_end
    return label


@parse_action(line_no)
def parse_line_no(toks):
    loc_start, (tok,), loc_end = toks
    lineno = LineNo(int(tok))
    lineno.loc_start = loc_start
    lineno.loc_end = loc_end
    return lineno


@parse_action(line)
def parse_line(toks):
    return Line(list(toks))


@parse_action(sub_stmt)
def parse_sub_stmt(toks):
    name, params, is_static = toks
    params = params or []
    return SubStmt(name, params, bool(is_static))


@parse_action(end_sub_stmt)
def parse_end_sub(toks):
    return EndSubStmt()


@parse_action(exit_sub_stmt)
def parse_exit_sub(toks):
    return ExitSubStmt()


@parse_action(function_stmt)
def parse_function_stmt(toks):
    name, params, is_static = toks
    params = params or []
    return FunctionStmt(name, params, bool(is_static))


@parse_action(end_function_stmt)
def parse_end_function(toks):
    return EndFunctionStmt()


@parse_action(exit_function_stmt)
def parse_exit_function(toks):
    return ExitFunctionStmt()


@parse_action(array_dim_range)
def parse_array_dim_range(toks):
    loc_start, (dim_range,), loc_end = toks
    if len(dim_range) == 1:
        lbound = NumericLiteral(0)
        ubound = dim_range[0]
    else:
        lbound, ubound = dim_range
    result = ArrayDimRange(lbound, ubound)
    result.loc_start = loc_start
    result.loc_end = loc_end
    return result


@parse_action(var_decl)
@parse_action(type_field_decl)
@parse_action(declare_var_decl)
@parse_action(param_decl)
def parse_var_decl(toks):
    loc_start, toks, loc_end = toks

    if len(toks) == 1:
        # This should be the case where param_decl is the same as a
        # normal var_decl (no empty "()")
        assert isinstance(toks[0], VarDeclClause)
        return toks[0]

    if len(toks) == 2:
        name, dims = toks
        type_name = None
    elif len(toks) == 3:
        # type_field_decl, which does not support array_indices
        name, _, type_name = toks
        dims = []
    elif len(toks) == 4:
        name, dims, _, type_name = toks
    else:
        assert False

    dims = dims or []
    dims = list(dims)
    if dims == ['(', ')']:
        is_nodim_array = True
        dims = []
    else:
        is_nodim_array = False

    # This should have been done in the parse_identifier, but that
    # parse action is not always called at the moment due to a bug in
    # pyparsing
    name = name.lower()

    if type_name == 'any':
        clause = AnyVarDeclClause(name)
    else:
        clause = VarDeclClause(name, type_name, dims,
                               is_nodim_array=is_nodim_array)

    clause.loc_start = loc_start
    clause.loc_end = loc_end

    return clause


@parse_action(type_stmt)
def parse_type_stmt(toks):
    return TypeStmt(toks[0])


@parse_action(end_type_stmt)
def parse_end_type(toks):
    return EndTypeStmt()


@parse_action(view_print_stmt)
def parse_view_print(toks):
    if toks[0] is None:
        return ViewPrintStmt(None, None)

    top_expr, bottom_expr = toks[0]
    return ViewPrintStmt(top_expr, bottom_expr)


@parse_action(while_stmt)
def parse_while_stmt(toks):
    cond = toks[0]
    return WhileStmt(cond)


@parse_action(wend_stmt)
def parse_wend_stmt(toks):
    return WendStmt()


@parse_action(width_stmt)
def parse_width_stmt(toks):
    if len(toks) == 3:
        columns, _, lines = toks
    elif len(toks) == 2:
        _, lines = toks
        columns = None
    else:
        columns = toks[0]
        lines = None
    return WidthStmt(columns, lines)


def main():
    parser = argparse.ArgumentParser(
        description='A script used for testing the grammar.')

    parser.add_argument('expr', nargs='?', help='The value to parse.')
    parser.add_argument(
        '--file', '-f',
        help='The file to read the value to parse from.')
    parser.add_argument(
        '--rule', '-r', default='line',
        help='The rule to use for parsing. Defaults to "%(default)s".')
    parser.add_argument(
        '--not-all', '-n', action='store_true',
        help='If set, the parse_all flag is set to false.')
    parser.add_argument(
        '--tree', '-t', action='store_true',
        help='If set, draws the AST as a tree.')

    args = parser.parse_args()

    if not args.expr and not args.file:
        print('Either specify an expression to parse or use --file.')
        exit(1)

    if args.expr and args.file:
        print('Cannot use --file together with an expression to parse.')
        exit(1)

    if args.file:
        with open(args.file) as f:
            args.expr = f.read()

    rule = globals().get(args.rule)
    if not isinstance(rule, ParserElement):
        print(f'No such rule found: {args.rule}')
        exit(1)

    try:
        result = rule.parse_string(args.expr,
                                   parse_all=not args.not_all)
    except (ParseException, ParseSyntaxException) as e:
        print(e.explain())
        exit(1)

    if args.tree:
        draw_tree(result[0])
    else:
        if hasattr(result, '__len__') and \
           len(result) and \
           isinstance(result[0], Line):
            for node in result[0].nodes:
                print(node)
        else:
            print(result)


def draw_tree(node, depth=0):
    node_desc = type(node).__name__
    if depth == 0:
        print(node_desc)
    else:
        print('|' * (depth - 1) + '+' + node_desc)
    for child in node.children:
        draw_tree(child, depth + 1)


if __name__ == '__main__':
    main()
