import sys
from mitim_tools.experiment_tools import AUGtools

fileASTRA = sys.argv[1]
folderUFs = sys.argv[2]

AUGtools.createFolderFromASTRA(fileASTRA, folderUFs)
