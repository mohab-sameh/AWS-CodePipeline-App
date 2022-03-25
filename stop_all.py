import os
import signal
import subprocess
import sys
import time



def signal_handler(signal_number, stack_frame):
    for process in processes:
        process.kill()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if sys.platform == 'win32':
    signal.signal(signal.SIGBREAK, signal_handler)
else:
    signal.signal(signal.SIGQUIT, signal_handler)
