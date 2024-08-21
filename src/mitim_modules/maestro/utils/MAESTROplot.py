import os
from mitim_modules.maestro.MAESTROmain import maestro

def plotMAESTRO(folder, num_beats = 2, only_beats = None):

    # Find beat results from folders
    folder_beats = f'{folder}/Beats/'
    beats = sorted(os.listdir(folder_beats))

    beat_types = [] 
    for beat in range(len(beats)):
        if 'run_transp' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('transp')
        elif 'run_portals' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('portals')
            
    # Create "dummy" maestro by only defining the beats
    m = maestro(folder)
    for beat in beat_types:
        m.define_beat(beat)

    # Plot
    m.plot(num_beats=num_beats, only_beats = only_beats)

    return m
