import numpy as np
import sys 
from glob import glob
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import seaborn as sns

from scipy.signal import savgol_filter

plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

sns.set_style("whitegrid")

# Tableau 20 Colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
             
# Tableau Color Blind 10
tableau20blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (200, 82, 0), (137, 137, 137), (163, 200, 236),
             (255, 188, 121), (207, 207, 207)]
  
# Rescale to values between 0 and 1 
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)
    
for i in range(len(tableau20blind)):  
    r, g, b = tableau20blind[i]  
    tableau20blind[i] = (r / 255., g / 255., b / 255.)

markers = ['+', 'x', '.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]

patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.', '/')

dataset = sys.argv[1]

seed_list = [42, 1234, 2011]
# seed_list = [42]

algos = ["PULSE", "kPU", "domain_disc", "backpropODA"]


aggregate_results = {}

for alg in algos:
    filename = f"outputs_final/{dataset}/{alg}_*_*_%d_log.txt"
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

        # print(seed_results.shape)
        if alg not in aggregate_results:
            aggregate_results[alg]  = [seed_results]
        else:
            aggregate_results[alg].append(seed_results)

mean_results = {}
var_results = {}

win_size = 3
for alg in aggregate_results:
    for i, seed in enumerate(seed_list): 
        print(aggregate_results[alg][i].shape)
        aggregate_results[alg][i] = savgol_filter(aggregate_results[alg][i][:, :3], win_size, 1, axis=0)

    mean_results[alg] = np.mean(aggregate_results[alg], axis=0)
    var_results[alg] = np.var(aggregate_results[alg], axis=0)


fig,ax = plt.subplots()

l=3.0
fc=20

ax.plot(mean_results["backpropODA"][:,0], mean_results["backpropODA"][:,2], linewidth=l, color=tableau20blind[0], label="BODA")
ax.fill_between(mean_results["backpropODA"][:,0], mean_results["backpropODA"][:,2] - var_results["backpropODA"][:,2], mean_results["backpropODA"][:,2] + var_results["backpropODA"][:,2], color=tableau20blind[0], alpha = 0.3)

ax.plot(mean_results["domain_disc"][:,0], mean_results["domain_disc"][:,2], linewidth=l, color=tableau20blind[1], label="Domain Disc")
ax.fill_between(mean_results["domain_disc"][:,0], mean_results["domain_disc"][:,2] - var_results["domain_disc"][:,2], mean_results["domain_disc"][:,2] + var_results["domain_disc"][:,2], color=tableau20blind[1], alpha = 0.3)

ax.plot(mean_results["kPU"][:, 0], mean_results["kPU"][:,2], linewidth=l, color=tableau20blind[2], label="kPU")
ax.fill_between(mean_results["kPU"][:, 0], mean_results["kPU"][:,2] - var_results["kPU"][:,2], mean_results["kPU"][:,2] + var_results["kPU"][:,2], color=tableau20blind[2], alpha = 0.3)

ax.plot(mean_results["PULSE"][:,0], mean_results["PULSE"][:,1], linewidth=l, color=tableau20blind[3], label="PULSE (Ours)")
ax.fill_between(mean_results["PULSE"][:, 0], mean_results["PULSE"][:,1] - var_results["PULSE"][:,1], mean_results["PULSE"][:,1] + var_results["PULSE"][:,1], color=tableau20blind[3], alpha = 0.3)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# plt.axvline(x=100, linestyle='--', linewidth=l)
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_xlabel('Epochs',fontsize=20)
plt.legend(prop={"size":18})
ax.legend()

plt.grid()
plt.savefig(f"plots/{dataset}_acc.png" ,transparent=True,bbox_inches='tight')
plt.clf()