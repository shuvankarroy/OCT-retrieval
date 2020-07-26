import os
import shutil
root = ".\\Duke\\"
for path, subdirs, files in os.walk(root): #list all files, directories in the path
    for file in files:
        # print(path)
        fullPath = path + "\\" + file
        
        if ("TIFFs" in fullPath) and ("8bitTIFFs" in fullPath):
            # print(fullPath)
            src = fullPath
            dest = path.split("\\")[:-2]
            dest = "\\".join(dest)
            
            shutil.move(src, dest)
            if file == files[-1]:
                folder = path.split("\\")[:-1]
                folder = "\\".join(folder)
                print(folder)
                shutil.rmtree(folder)
    