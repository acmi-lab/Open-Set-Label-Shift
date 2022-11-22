import os 
import sys
import shutil

folder = sys.argv[1]

for dir in ["benign", "malignant"]:
    curr_dir= f"{folder}/BreaKHis_v1/histology_slides/breast/{dir}/SOB/"
    
    for tumor_type in os.listdir(curr_dir):
        # Move tumor type dir to parent dir
        os.rename(f"{curr_dir}/{tumor_type}", f"{folder}/{tumor_type}_{dir}")
        
        
shutil.rmtree(f"{folder}/BreaKHis_v1")
       
    