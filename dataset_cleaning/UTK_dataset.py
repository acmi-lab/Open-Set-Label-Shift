import os
import sys 
import shutil 

source_dir= f"{sys.argv[1]}/UTKFace/"
target_dir = sys.argv[1]


# Walk through the directory
for root, dirs, files in os.walk(source_dir):
    print(root)
    for file_name in files: 
        print(file_name)
        age = int(file_name.split("_")[0])

        dir_name = age//10 
        
        if dir_name > 7: 
            dir_name= 7       

        # Create dir in target 
        if not os.path.exists(f"{target_dir}/{dir_name}"):
            os.makedirs(f"{target_dir}/{dir_name}")
        
        # Move file to target
        os.rename(f"{root}/{file_name}", f"{target_dir}/{dir_name}/{file_name}")

shutil.rmtree(source_dir)