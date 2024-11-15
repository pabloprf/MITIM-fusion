import re
import os
import sys
import datetime
import warnings
import contextlib
import logging
from IPython import embed

# Paramiko shows some deprecation warnings that are not relevant
# https://github.com/paramiko/paramiko/issues/2419
warnings.filterwarnings(action='ignore', module='.*paramiko.*')
logging.getLogger("paramiko").setLevel(logging.WARNING)

# Suppress only the "divide by zero" warning (PROFILEStools relevant)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

def printMsg(*args, typeMsg=""):
    """
    Print messages with different colors (blue-red is better for colorblind)
    It also accounts for verbosity
    """

    # Take into account the verbose level
    from mitim_tools.misc_tools.CONFIGread import read_verbose_level

    verbose = read_verbose_level()

    if verbose == 0:
        return False
    else:

        # -----------------------------------------------------------------------------
        # Define colors
        # -----------------------------------------------------------------------------

        # Info (about a choice or something found): Blue
        if typeMsg == "i":
            extra = "\u001b[34m"
        # Warning (about something to be careful about, even if chosen): Red
        elif typeMsg == "w":
            extra = "\u001b[31;1m"
        # Question or something that is stopped
        elif typeMsg == "q":
            extra = "\u001b[44;1m\u001b[37m"
        # Note: Nothing
        else:
            extra = "\u001b[0m"

        total = (extra,) + args + ("\u001b[0m",)

        # -----------------------------------------------------------------------------
        # Define action based on verbose
        # -----------------------------------------------------------------------------

        if verbose == 1:
            # Print
            if typeMsg in ["w"]:
                print(*total)
            # Question result
            return False

        elif verbose == 2:
            # Print
            if typeMsg in ["w", "q"]:
                print(*total)
            # Question result
            if typeMsg == "q":
                return query_yes_no("\t\t>> Do you want to continue?", extra=extra)

        elif verbose in [3,4]:
            # Print
            if typeMsg in ["w", "q", "i"]:
                print(*total)
            # Question result
            if typeMsg == "q":
                return query_yes_no("\t\t>> Do you want to continue?", extra=extra)

        elif verbose == 5:
            # Print
            print(*total)
            # Question result
            if typeMsg == "q":
                return query_yes_no("\t\t>> Do you want to continue?", extra=extra)


if not sys.platform.startswith('win'):
    import termios
    import tty

class prompting_context:
    def __init__(self):
        # For Unix-based systems, save the terminal settings
        if not sys.platform.startswith('win'):
            self.old_settings = termios.tcgetattr(sys.stdin)

    def __enter__(self):
        # Set raw mode for Unix-based systems
        if not sys.platform.startswith('win'):
            tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, *args):
        # Restore original terminal settings for Unix-based systems
        if not sys.platform.startswith('win'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self, prompt="Press a key: "):
        # Use input() for Windows; it requires pressing Enter
        if sys.platform.startswith('win'):
            print(prompt, end='', flush=True)
            key = input()[0]  # Capture only the first character
        else:
            # For Unix-based systems, read a single character
            key = sys.stdin.read(1)
        return key

def query_yes_no(question, extra=""):
    '''
    From https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input 
    '''

    valid = {"y": True, "n": False, "e": None}
    prompt = " [y/n/e] (yes, no, exit)"

    while True:
        total = (extra,) + (question,) + (prompt,) + ("\u001b[0m",)
        printMsg(*total)
        with prompting_context() as context:
            choice = context.get_key()
        if len(choice) > 1:
            choice = choice[0]
        if choice in valid:
            printMsg(f"\t\t>> Received answer: {choice}")
            if valid[choice] is not None:
                printMsg(
                    f'\t\t>> Proceeding sending "{valid[choice]}" flag to main program'
                )
                return valid[choice]
            else:
                raise Exception("[mitim] Exit request")
        else:
            printMsg("Please respond with 'y' (yes) or 'n' (no)\n")

class HiddenPrints:
    """
    Usage:
            with IOtools.HiddenPrints():
                    printMsg("This will not be printed")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

'''
Log file utilities
--------------------------------
chatGPT 4o as of 08/18/2024
'''

@contextlib.contextmanager
def conditional_log_to_file(log_file=None, msg=None):

    if log_file is not None:
        with log_to_file(log_file, msg) as logger:
            yield logger
    else:
        if msg:
            print(msg)  # Optionally print the message even if not logging to file
        yield None  # Simply pass through without logging

def strip_ansi_codes(text):
    if not isinstance(text, (str, bytes)):
        text = str(text)  # Convert non-string types to string
    # Strip ANSI escape codes
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)

class log_to_file:
    _context_count = 0  # Class attribute to keep track of context depth

    def __init__(self, log_file, msg=None):
        if msg is not None:
            print(msg)
        self.log_file = log_file
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log = None
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None

    def __enter__(self):
        if log_to_file._context_count == 0:
            # First entry into the context, set up logging
            self.log = open(self.log_file, 'a')
            self.stdout_fd = sys.stdout.fileno()
            self.stderr_fd = sys.stderr.fileno()

            # Save the actual stdout (1) and stderr (2) file descriptors.
            self.saved_stdout_fd = os.dup(self.stdout_fd)
            self.saved_stderr_fd = os.dup(self.stderr_fd)

            # Redirect stdout and stderr to the log file.
            os.dup2(self.log.fileno(), self.stdout_fd)
            os.dup2(self.log.fileno(), self.stderr_fd)

            # Redirect Python's sys.stdout and sys.stderr to the log file.
            sys.stdout = self
            sys.stderr = self

            # Redirect warnings to the log file
            def logging_handler(message, category, filename, lineno, file=None, line=None):
                self.log.write(f"{category.__name__}: {strip_ansi_codes(message)}\n")
            warnings.showwarning = logging_handler

        log_to_file._context_count += 1  # Increment the context depth

        return self

    def write(self, message):
        # Remove ANSI codes from the message before writing to the log
        clean_message = strip_ansi_codes(message)
        try:
            self.log.write(clean_message)
        except ValueError:
            # If the file is closed, reopen it and try again
            self.log = open(self.log_file, 'a')
            self.log.write(clean_message)
        self.log.flush()  # Ensure each write is immediately flushed

    def flush(self):
        # Ensure sys.stdout and sys.stderr are flushed
        self.log.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        log_to_file._context_count -= 1  # Decrement the context depth

        if log_to_file._context_count == 0:
            # Last exit from the context, restore the original state
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = self.stdout
            sys.stderr = self.stderr

            os.dup2(self.saved_stdout_fd, self.stdout_fd)
            os.dup2(self.saved_stderr_fd, self.stderr_fd)

            os.close(self.saved_stdout_fd)
            os.close(self.saved_stderr_fd)

            self.log.close()

            # Restore the original warnings behavior
            warnings.showwarning = warnings._showwarning_orig

        # If still inside a context, don't close the file or restore the state

class redirect_all_output_to_file:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        self.stdout_fd = None
        self.stderr_fd = None
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None
        self.logfile = None

    def __enter__(self):
        # Save the actual stdout and stderr file descriptors.
        self.stdout_fd = sys.__stdout__.fileno()
        self.stderr_fd = sys.__stderr__.fileno()

        # Save a copy of the original file descriptors.
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)

        # Open the log file.
        self.logfile = open(self.logfile_path, 'w')

        # Redirect stdout and stderr to the log file.
        os.dup2(self.logfile.fileno(), self.stdout_fd)
        os.dup2(self.logfile.fileno(), self.stderr_fd)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore stdout and stderr from the saved file descriptors.
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)

        # Close the duplicated file descriptors.
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)

        # Close the log file.
        if self.logfile:
            self.logfile.close()

def ignoreWarnings(module=None):
    if module is None:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        warnings.filterwarnings("ignore", module=module)  # "matplotlib\..*" )

'''
----------------------------------------------------------------------------------------
Logger
******
    This class is used to log the output of the program to a file. It is used in the
----------------------------------------------------------------------------------------
'''

class Logger(object):
    def __init__(self, logFile="logfile.log", DebugMode=0, writeAlsoTerminal=True):
        self.terminal = sys.stdout
        self.logFile = logFile
        self.writeAlsoTerminal = writeAlsoTerminal

        currentime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"- Creating log file: {logFile}")

        if DebugMode == 0:
            with open(self.logFile, "w") as f:
                f.write(f"* New run ({currentime})\n")
        else:
            with open(self.logFile, "a") as f:
                f.write(
                    f"\n\n\n\n\n\t ~~~~~ Run cold_started ({currentime})~~~~~ \n\n\n\n\n"
                )

    def write(self, message):
        if self.writeAlsoTerminal:
            self.terminal.write(message)

        with open(self.logFile, "a") as self.log:
            self.log.write(strip_ansi_codes(message))

    # For python 3 compatibility:
    def flush(self):
        pass
