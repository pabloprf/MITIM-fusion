from cProfile import label
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class GKplotting:
    def _plot_trace(self, ax, object_or_label, variable, c="b", lw=1, ls="-", label_plot='', meanstd=True, var_meanstd= None):

        if isinstance(object_or_label, str):
            object_grab = self.results[object_or_label]
        else:
            object_grab = object_or_label

        t = object_grab.t
        
        if not isinstance(variable, str):
            z = variable
            if var_meanstd is not None:
                z_mean = var_meanstd[0]
                z_std = var_meanstd[1]
            
        else:
            z = object_grab.__dict__[variable]
            if meanstd and (f'{variable}_mean' in object_grab.__dict__):
                z_mean = object_grab.__dict__[variable + '_mean']
                z_std = object_grab.__dict__[variable + '_std']
            else:
                z_mean = None
                z_std = None
        
        ax.plot(
            t,
            z,
            ls=ls,
            lw=lw,
            c=c,
            label=label_plot,
        )
        
        if meanstd and z_std>0.0:
            GRAPHICStools.fillGraph(
                ax,
                t[t>object_grab.tmin],
                z_mean,
                y_down=z_mean
                - z_std,
                y_up=z_mean
                + z_std,
                alpha=0.1,
                color=c,
                lw=0.5,
                islwOnlyMean=True,
                label=label_plot + f" $\\mathbf{{{z_mean:.3f} \\pm {z_std:.3f}}}$ (1$\\sigma$)",
            )