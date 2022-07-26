import numpy as np
import sys 
from glob import glob

dataset = sys.argv[1]
alg = sys.argv[2]

if len(sys.argv) > 3:
    keep_last = int(sys.argv[3])
else:
    keep_last = 1

seed_list = [42, 1234, 2011]
# seed_list = [42]

filename = f"outputs/{dataset}/{alg}_*_28_%d_log.txt"

aggregate_results = []

for seed in seed_list: 
    seed_results = []
    curr_filename = filename % seed
    print(curr_filename)

    curr_files = glob(curr_filename)

    # curr_files = [curr_filename]
    
    if len(curr_files) == 0:
        print("No files found")
        exit()
    elif len(curr_files) == 1:
        for curr_file in curr_files: 
            with open(curr_file, 'r') as f: 
                lines = f.readlines()
                for line in lines:
                    line.rstrip() 
                    temp_arr =[]
                    arr = line.split(',')
                    for val in arr: 
                        if val.strip() != "NA": 
                            temp_arr.append(float(val))
                        else: 
                            temp_arr.append(0)

                    seed_results.append(temp_arr)
        
        seed_results = np.array(seed_results)
        
    else:
        print("More than one file found! ")
        print(curr_files)
        exit()

    for i in range(keep_last): 
        aggregate_results.append(seed_results[-i-1])


aggregate_results = np.array(aggregate_results)

print(np.mean(aggregate_results, axis=0))
