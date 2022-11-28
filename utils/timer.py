import time

class Timer:
    '''
    Class to model a timer with partial interval capabilities.
    '''
    
    since: float = 0
    end: float = 0
    delta: float = 0
    buffer_size: int = 10
    running_pointer: int = 0
    partials: list = []
    
    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.reset()
        
    def start(self):
        self.since = time.time()
    
    def partial(self):
        partial = time.time() - self.since
        self.partials[self.running_pointer] = partial
        self.__increment_pointer()
        return partial
    
    def stop(self):
        self.end = time.time()
        self.delta = self.end - self.since
        return self.delta
        
    def reset(self):
        self.since = 0
        self.end = 0
        self.delta = 0
        self.running_pointer = 0
        self.partials = [0 for i in range(self.buffer_size)] 
        
    def __increment_pointer(self):
        self.running_pointer += 1
        if self.running_pointer >= self.buffer_size:
            self.running_pointer = 0