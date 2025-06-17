import time


class PerformaceTimer:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()

    def next(self, name):
        print(f"{self.name} time ms: ", (time.time() - self.start_time) * 1000)
        self.start_time = time.time()
        self.name = name