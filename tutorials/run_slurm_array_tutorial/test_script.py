import sys
import os

folder = str(sys.argv[1])
# print('Folder:', folder)    


i = str(sys.argv[2])
# print('i:', i)

with open(f"{folder}/file_successfully_created_{i}.txt", "w") as f:
    input_text = 'Successfully created file from job #:' + i
    f.write(f"{input_text}\n")