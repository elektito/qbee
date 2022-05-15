from .stmt import Stmt
from .program import Label
from .exceptions import InternalError, CodeGenError


class BaseCode:
    pass


class CodeGen:
    # maps names of codegens to codegen classes
    codegens = {}

    def __init__(self, codegen_name, compiler):
        self.compiler = compiler
        self.codegen_name = codegen_name

        if self.codegen_name not in CodeGen.codegens:
            raise CodeGenError(
                f'Unknown code generator: {self.codegen_name}')

        impl_class = CodeGen.codegens[codegen_name]
        self._impl = impl_class(compiler)

    def gen_code(self, program):
        return self._impl.gen_code(program)


class CodeGenMetaclass(type):
    """Metaclass for all code generator classes, applied through the base
class BaseCodeGen. It expects two arguments, 'cg_name' and 'code_class',
to be passed to the class.

The following attributes are added to classes:

 - name: the value of the 'cg_name' keyword argument
 - code_class: the value of the 'code_class' keyword argument
 - generator_funcs: A dictionary which will be populated by
   BaseCodeGen.generator_for decorator and maps Node sub-classes to
   the generator functions declared with the decorator.

    """

    def __new__(cls, name, bases, attrs, **kwargs):
        if name == 'BaseCodeGen':
            return super().__new__(cls, name, bases, attrs)

        if 'cg_name' not in kwargs:
            raise InternalError(
                'CodeGen class does not specify a "cg_name". Pass a '
                'keyword argument "cg_name" to the class.')

        if 'code_class' not in kwargs:
            raise InternalError(
                'CodeGen class does not specify a "code_class". Pass a '
                'keyword argument "code_class" to the class.')

        codegen_name = kwargs['cg_name']
        if not isinstance(codegen_name, str):
            raise InternalError('CodeGen name must be a string')

        code_class = kwargs['code_class']
        if codegen_name in CodeGen.codegens:
            raise InternalError(
                f'Duplicate CodeGen name: {codegen_name}')

        code_class = kwargs['code_class']
        if not isinstance(code_class, type):
            raise InternalError(
                'The value of "code_class" argument must be a "type".')
        if not issubclass(code_class, BaseCode):
            raise InternalError(
                f'{code_class.__name__} is not a subclass of '
                f'codegen.BaseCode')

        attrs['name'] = codegen_name
        attrs['code_class'] = code_class
        attrs['generator_funcs'] = {}
        klass = super().__new__(cls, name, bases, attrs)

        CodeGen.codegens[codegen_name] = klass

        return klass


class BaseCodeGen(metaclass=CodeGenMetaclass):
    "Base class for all code generator classes."

    @classmethod
    def generator_for(cls, node_class):
        def decorator(func):
            if node_class in cls.generator_funcs:
                raise InternalError(
                    f'Duplicate codegen function for type: '
                    f'{node_class.__name__}')
            cls.generator_funcs[node_class] = func
            return func
        return decorator

    def gen_code(self, program):
        code = self.code_class()
        for node in program.children:
            self.gen_code_for_node(node, code)
        return code

    def gen_code_for_node(self, node, code):
        """Calls the appropriate code generator functions for the given
node and adds the results to the given code object. The main purpose of
this method is to be called by codegen functions themselves to call
other codegen functions.

        """

        gen = self.generator_funcs.get(type(node))
        if not gen:
            raise InternalError(
                f'Cannot generate code for node: {node}')
        gen(node, code, self)
