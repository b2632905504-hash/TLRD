import sys

class DualWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for compatibility with the standard stdout interface.
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        # vLLM expects sys.stdout to expose fileno()
        return self.terminal.fileno()

    def isatty(self):
        # Preserve terminal behavior checks
        return self.terminal.isatty()
        
