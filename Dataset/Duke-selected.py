import os
import shutil
root = ".\\Duke\\"
dest_root = ".\\Duke-selected\\"
os.makedirs(".\\Duke-selected\\", exist_ok=True)

file_list = []
for i in range(1, 38):
	file_name = str(i)
	act_file_name = str(i).zfill(len(str(i))+1) + ".tif"
	file_list.append(act_file_name)


for path, subdirs, files in os.walk(root): #list all files, directories in the path
    for file in files:
        # print(path)
        fullPath = path + "\\" + file
        
        if file in file_list:
            print(fullPath)
            src = fullPath
            dest = path.replace(root, dest_root)
            
            print(dest)
            os.makedirs(dest + "\\", exist_ok=True)
            shutil.copy(src, dest+"\\")
        