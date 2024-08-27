"""
Quick way to check
e.g.
		run_check.py unsername1 unsername2
"""

import sys, time
from mitim_tools.transp_tools.src import TRANSPglobus

from mitim_tools.misc_tools.CONFIGread import read_verbose_level

def main():

    

    # User inputs
    users = sys.argv[1:]

    secondsWait = 30

    # Workflow
    while True:
        TRANSPglobus.printUser(users)
        print(f"- Next status check in {secondsWait}sec")
        time.sleep(secondsWait)

if __name__ == "__main__":
    main()
