from mitim_tools.astra_tools import ASTRAtools    

astra = ASTRAtools.ASTRA()

astra.prep('~/scratch/testAstra/')

#astra.expfile = ''

astra.run(1.0,1.005,name='run1')
