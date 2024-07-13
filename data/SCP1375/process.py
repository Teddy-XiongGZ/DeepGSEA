import pandas as pd
import numpy as np
import os

# data_dir = "../data/SCP1375"
data_dir = "./"

dat = pd.read_csv(os.path.join(data_dir, "expression_matrix_normalized.csv"), index_col=0)

with open(os.path.join(data_dir, "genes.txt"), 'w') as f:
    f.write('\n'.join([i.upper() for i in dat.index]))

np.save(os.path.join(data_dir, "expression_matrix_normalized.npy"), dat.values.T)