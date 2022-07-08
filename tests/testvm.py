from functools import partial
from qvm.machine import QvmMachine


class TestPeripheralsImpl:
    __test__ = False

    def __init__(self, test_case):
        self.io = []

        self.inkey_list = test_case.inkey_list
        self.inkey_idx = 0

        self.rnd_list = test_case.rnd_list
        self.rnd_idx = 0

        self.timer_list = test_case.timer_list
        self.timer_idx = 0

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

    def rng_get_next(self):
        idx = self.rnd_idx
        self.rnd_idx += 1
        return self.rnd_list[idx]

    def rng_get_with_seed(self, seed):
        return abs(seed / 100)

    def time_get_time(self):
        idx = self.timer_idx
        self.timer_idx += 1
        return self.timer_list[idx]

    def terminal_inkey(self):
        idx = self.inkey_idx
        self.inkey_idx += 1
        if idx >= len(self.inkey_list):
            return ''
        return self.inkey_list[idx]


class TestMachine:
    __test__ = False

    def init(self, module, test_case):
        self.impl = TestPeripheralsImpl(test_case)
        self._machine = QvmMachine(module, impl=self.impl)

    def run(self):
        self._machine.run()

    @property
    def cpu(self):
        return self._machine.cpu

    @property
    def io(self):
        return self.impl.io
