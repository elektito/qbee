from multiprocessing import Process, Queue
from queue import Empty


class SubTerminal:
    def __init__(self):
        self.request_queue = Queue(512)
        self.result_queue = Queue(512)

    def launch(self):
        self.process = Process(target=self.run, daemon=True)
        self.process.start()

    def run(self):
        # We have to import these here, in the child process,
        # otherwise pyglet will not be able to create an OpenGL
        # context (since OpenGL context has already been initialized
        # in the parent process)
        import pyglet
        from qvm.terminal import TerminalWindow

        self.window = TerminalWindow(
            request_queue=self.request_queue,
            result_queue=self.result_queue,
            caption='QVM Terminal',
            resizable=True,
        )
        pyglet.app.run()

    def call(self, method_name, *args, **kwargs):
        self.request_queue.put((method_name, args, kwargs, False))

    def call_with_result(self, method_name, *args, **kwargs):
        self.request_queue.put((method_name, args, kwargs, True))
        return self.result_queue.get()
