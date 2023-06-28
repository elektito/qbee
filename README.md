# qbee 🐝

*qbee* is a QBASIC compiler written in Python. The compilation target
is a virtual machine called QVM. Most of the core language is
complete, but graphics routines are mostly missing. qbee can compile
and run the classic nibbles example:

```
$ python -m qbee.main -o nibbles.mod -O3 -v nibbles.bas
$ python -m qvm.run nibbles.mod
```

## Debugger
qbee has a source-level debugger capable of tracing code, setting
breakpoints and evaluating variables and expressions at run-time. In
order to be able to use it, you need to compile the source using the
`-g` compiler option.

```
$ python -m qbee.main -o nibbles.mod -O3 -v -g nibbles.bas
```

Then you can run the compiled module in the debugger:

```
python -m qvm.dbg nibbles.mod
```

You can step through the program using `next` and `step` commands, set
breakpoints using `break`, and evaluate expressions using `print`. Run
`help` to get a full list of all available debugger commands.

## QVM
You can find a description of the QVM virtual machine, and its
instruction set at the [project wiki][1] on github.

Here's a short example so that you can see what the QVM assembly looks
like:

```
$ cat >foo.bas <<EOF
> defint a-z
> input i
> if i < 10 then
>    print "too small"
> else
>    print "too big"
> end if
> EOF

$ python -m qbee.main -O3 --asm -o- foo.bas
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.literals
    0 string ""
    1 string "too small"
    2 string "too big"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.routines

_main:
    integer i

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.code

    call        _sub__main
    halt
_sub__main:
    frame       0, 1
    push0%
    push$       ""
    pushm1%
    push1%
    push1%
    io          terminal, input
    storel      i
    readl%      i
    push%       10
    cmp
    lt
    jz          _else_2
    push0%
    push$       "too small"
    push2%
    io          terminal, print
    jmp         _endif_1
_else_2:
    push0%
    push$       "too big"
    push2%
    io          terminal, print
_endif_1:
    ret
```

## Tests
`qbee` has a fairly extensive test suite. You can run the tests using
pytest: `python -m pytest`

## Compatibility
Python 3.9 or higher is required. This is needed because:

 - Python 3.9 allows having `classmethod` and `property` decorators on
   the same method.
 - It allows using indexed built-in types for type hinting (like
   `list[int]`, instead of `List[int]`)

[1]: https://github.com/elektito/qbee/wiki/QVM
