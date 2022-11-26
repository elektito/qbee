# qbee :honeybee:

*qbee* is a QBASIC compiler written in Python. The compilation target
is a virtual machine called QVM. Most of the core language is
complete, but graphics routines are mostly missing. qbee can compile
and run the classic nibbles example:

```
$ python -m qbee.main -o nibbles.mod -O3 -v nibbles.bas
$ python -m qvm.run nibbles.mod
```

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

`qbee` has a fairly extensive test suite. You can run the tests using
pytest: `python -m pytest`

Python 3.9 or higher is required. This is needed because:

 - Python 3.9 allows having `classmethod` and `property` decorators on
   the same method.
 - It allows using indexed built-in types for type hinting (like
   `list[int]`, instead of `List[int]`)
