import os
from IPython import embed
from mitim_modules.maestro import MAESTROmain

def plotMAESTRO(folder, num_beats = 2):

    folder_beats = f'{folder}/Beats/'
    beats = sorted(os.listdir(folder_beats))

    beat_types = [] 
    how_many = min(num_beats, len(beats))
    for beat in range(how_many):
        if 'run_transp' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('transp')
        elif 'run_portals' in os.listdir(f'{folder_beats}/{beats[beat]}'):
            beat_types.append('portals')
            
    m = MAESTROmain.maestro(folder)
    for beat in beat_types:
        m.define_beat(beat)

    m.plot()

    return m
