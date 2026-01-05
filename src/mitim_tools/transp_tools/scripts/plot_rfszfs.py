import argparse
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.misc_tools import GRAPHICStools
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

def plot_rfszfs(rfs_file, zfs_file, ax = None, time_pos = -1, surf_pos = -1, c='b'):
    
    print(f"Plotting RFS from {rfs_file} and ZFS from {zfs_file}")
    
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))
        
    # Load data
    rfs = UFILEStools.UFILEtransp()
    rfs.readUFILE(rfs_file)
    R = rfs.Variables['Z'][time_pos, :, surf_pos]
    
    zfs = UFILEStools.UFILEtransp()
    zfs.readUFILE(zfs_file)
    Z = zfs.Variables['Z'][time_pos, :, surf_pos]
    
    # Plot
    ax.plot(R, Z, '-o', c=c, markersize=2, lw=0.5, label=f'Time {time_pos}, Surf {surf_pos}')
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    ax.axis('equal')
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("prefix", type=str)

    args = argparser.parse_args()
    rfs_file = args.prefix + ".RFS"
    zfs_file = args.prefix + ".ZFS"
    
    # Capture major and minor radius (geometric)
    rfs = UFILEStools.UFILEtransp()
    rfs.readUFILE(rfs_file)
    
    R, a = [], []
    for time_pos in range(rfs.Variables['Z'].shape[0]):
    
        R.append( np.mean([rfs.Variables['Z'][time_pos,:,-1].max(),rfs.Variables['Z'][time_pos,:,-1].min()]))
        a.append( 0.5*(rfs.Variables['Z'][time_pos,:,-1].max() - rfs.Variables['Z'][time_pos,:,-1].min()) )
        
    fig, axs = plt.subplots(ncols=2, figsize=(12,6))
    cols = GRAPHICStools.listColors()
    cont = 0
    for time_pos in range(rfs.Variables['Z'].shape[0]):
        for surf_pos in range(rfs.Variables['Z'].shape[2]):
            plot_rfszfs(rfs_file, zfs_file, ax=axs[0], time_pos=time_pos, surf_pos=surf_pos, c=cols[cont])
            
            cont += 1   
            
    axs[1].plot(R, label='R (m)')
    axs[1].plot(a, label='a (m)')
    axs[1].set_xlabel('Time index')
    axs[1].set_ylabel('m')
    axs[1].legend()
    GRAPHICStools.addDenseAxis(axs[1])
    axs[1].set_title('Geometric major and minor radius evolution (Last surface)')

                
    plt.show()
    embed()
