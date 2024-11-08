from mitim_tools.astra_tools import ASTRAtools 
from mitim_tools import __mitimroot__   

astra = ASTRAtools.ASTRA()

folder = __mitimroot__ / "tests" / "scratch" / "astra_test"

astra.prep(folder)

#astra.expfile = ''

astra.run(1.0,1.005,name='run1')
