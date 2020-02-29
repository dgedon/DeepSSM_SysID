# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
import sys
import os


def set_redirects(logdir, file_name):
    sys.stdout = Logger(logdir, file_name, sys.stdout)
    sys.stderr = Logger(logdir, file_name, sys.stderr)


class Logger(object):
    def __init__(self, logdir, file_name, std):
        self.terminal = std
        self.log = open(os.path.join(logdir, file_name + '.log'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

