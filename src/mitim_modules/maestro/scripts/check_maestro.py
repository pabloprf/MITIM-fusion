from pathlib import Path
import argparse
import re

colors = {
    "PORTALS": "\033[31m",   # red
    "EPED": "\033[35m",      # green
    "TRANSP": "\033[33m",    # yellow
    "UNKNOWN": "\033[34m",   # blue
    "FINISHED": "\033[32m",  # magenta
}
RESET = "\033[0m"

parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="*")

args = parser.parse_args()

folders = args.folders

folders_clean = []
for i in range(len(folders)):
    folders[i] = Path(folders[i])


    if '*' in folders[i].name:
        folders_all = list((folders[i].parent).iterdir())
        for f in folders_all:
            if folders[i].name[:-1] in f.name:
                folders_clean.append(f)
    else:
        folders_clean.append(folders[i])
folders = folders_clean

# Prepare list to hold each row of output as a tuple of strings
rows = [("Folder", "Last Beat", "Type", "Details")]

for folder in folders:

    beats_folder = Path(folder) / 'Beats'

    

    if not beats_folder.exists():
        rows.append((str(folder), "NO BEATS", "", ""))
        continue

    # Create a regex pattern to extract the numeric part
    pattern = re.compile(r'Beat_(\d+)')
    # List all subfolders that match the expected pattern
    subfolders = [d for d in beats_folder.iterdir() if d.is_dir() and pattern.match(d.name)]
    # Sort subfolders using the numeric part extracted by the regex
    sorted_subfolders = sorted(subfolders, key=lambda d: int(pattern.match(d.name).group(1)))

    last_beat = sorted_subfolders[-1]

    run_folder = [n.name for n in last_beat.iterdir()]


    outputs_folder = Path(folder) / 'Outputs'

    if (outputs_folder / 'beat_final').exists():

        # Grab how long it took
        slurm_output = Path(folder) / 'slurm_output.dat'
        with open(slurm_output, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if '* MAESTRO took' in line:
                txt = line.split('* MAESTRO took')[-1].strip()



        rows.append((str(folder), "FINISHED",last_beat.name, txt))
        continue


    txt = ''
    if 'run_portals' in run_folder:
        beat = 'PORTALS'

        exe_folder = last_beat / 'run_portals' / 'Execution'

        if not exe_folder.exists():
            txt = 'no execution folder'
        else:

            # Create a regex pattern to extract the numeric part
            pattern = re.compile(r'Evaluation.(\d+)')
            # List all subfolders that match the expected pattern
            subfolders = [d for d in exe_folder.iterdir() if d.is_dir() and pattern.match(d.name)]
            # Sort subfolders using the numeric part extracted by the regex
            sorted_subfolders = sorted(subfolders, key=lambda d: int(pattern.match(d.name).group(1)))

            last_ev = sorted_subfolders[-1].name.split('.')[-1]
            txt = f"last evaluation: {last_ev}"
    elif 'run_eped' in run_folder:
        beat = 'EPED'
    elif 'run_transp' in run_folder:
        beat = 'TRANSP'
    else:
        beat = 'UNKNOWN'
    
    # Save output as tuple: (folder, last beat name, beat type, details)
    rows.append((str(folder), last_beat.name, beat, txt))



# Determine maximum width for each column
col_widths = [0, 0, 0, 0]
for row in rows:
    for i, col in enumerate(row):
        col_widths[i] = max(col_widths[i], len(col))

# Print each row with columns aligned and colored based on beat type.
for i,row in enumerate(rows):
    beat_type = row[2] if row[2] else ("FINISHED" if row[1] == "FINISHED" else "UNKNOWN")
    color = colors.get(beat_type, "")
    line = f"{row[0]:<{col_widths[0]}} - {row[1]:<{col_widths[1]}} - {row[2]:<{col_widths[2]}} - {row[3]}"
    print(f"{color}{line}{RESET}")
    if i == 0:
        print("-" * 50)
print('')