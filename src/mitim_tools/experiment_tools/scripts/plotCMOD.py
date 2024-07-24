from mitim_tools.experiment_tools import CMODtools

c = CMODtools.experiment(1150903021)

c.get2Dprofiles()
c.slice2Dprofiles(1.0, avt=0.2)
c.plot2Dprofiles()
