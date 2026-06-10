from pathlib import Path
import argparse
from mitim_tools.gacode_tools.utils import GACODErun


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TRXPL postprocessing on GACODE CDF outputs.")
    parser.add_argument("folder",type=str,help="Working folder where GACODE outputs are located.")
    parser.add_argument("time",type=float,help="Time in seconds to extract")
    parser.add_argument("--avTime",type=float,default=0.0,help="Averaging time window in seconds.")
    parser.add_argument("--bt", type=int, default=0, help="Direction of Bt")
    parser.add_argument("--ip", type=int, default=0, help="Direction of Ip")
    parser.add_argument("--grids",nargs=3,type=int,default=[151,101,101],help="Grid points")
    
    
    args = parser.parse_args()
    folder = Path(args.folder)
    timeRun = args.time
    avTime = args.avTime
    BtIp_dirs = [args.bt, args.ip]
    grids = args.grids
    
    for item in folder.glob('*'):
        if "TR.DAT" in item.name:
            nameFiles = item.name[:-"TR.DAT".__len__()]
            break

    print(f"Executing TRXPL for {nameFiles}")

    GACODErun.runTRXPL(
        folder,
        timeRun,
        BtDir=BtIp_dirs[0],
        IpDir=BtIp_dirs[1],
        avTime=avTime,
        nameFiles=nameFiles,
        sendState=True,
        grids=grids,
    )