from functools import partial
from qvm.machine import QvmMachine


class TestPeripheralsImpl:
    __test__ = False

    def __init__(self):
        self.io = []

    def __getattr__(self, attr):
        devices = [
            'data', 'memory', 'pcspkr', 'rng', 'terminal', 'time',
        ]
        for d in devices:
            if attr.startswith(d + '_'):
                device_name = d
                op_name = attr[len(d) + 1:]
                return partial(self.any_io, device_name, op_name)
        raise AttributeError

    def any_io(self, device_name, op_name, *args):
        self.io.append((device_name, op_name, *args))


class TestMachine:
    __test__ = False

    def init(self, module):
        self.impl = TestPeripheralsImpl()
        self._machine = QvmMachine(module, impl=self.impl)

    def run(self):
        self._machine.run()

    @property
    def cpu(self):
        return self._machine.cpu

    @property
    def io(self):
        return self.impl.io
