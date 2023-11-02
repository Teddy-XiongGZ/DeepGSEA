import os
import numpy as np
import pandas as pd

# data_dir = "../data/GSE122031"
data_dir = "./"

group0 = pd.read_csv(os.path.join(data_dir, "GSM3453216_A549_4h_Mock_NormCounts.csv"), index_col=0).T # (4505, 25379)
group1 = pd.read_csv(os.path.join(data_dir, "GSM3453215_A549_4h_MOI02_NormCounts.csv"), index_col=0).T # (7377, 25379)
group2 = pd.read_csv(os.path.join(data_dir, "GSM3453214_A549_4h_MOI20_NormCounts.csv"), index_col=0).T # (6758, 25379)

group0 = group0.iloc[:,group0.columns.notna()]
group1 = group1.iloc[:,group1.columns.notna()]
group2 = group2.iloc[:,group2.columns.notna()]

with open(os.path.join(data_dir, "genes.txt"), 'w') as f:
    f.write('\n'.join([i.upper() for i in group0.columns]))

with open(os.path.join(data_dir, "barcodes_4h_mock.txt"), 'w') as f:
    f.write('\n'.join([i for i in group0.index]))

with open(os.path.join(data_dir, "barcodes_4h_moi02.txt"), 'w') as f:
    f.write('\n'.join([i for i in group1.index]))

with open(os.path.join(data_dir, "barcodes_4h_moi20.txt"), 'w') as f:
    f.write('\n'.join([i for i in group2.index]))

np.save(os.path.join(data_dir, "GSM3453216_A549_4h_Mock_NormCounts.npy"), group0.values)
np.save(os.path.join(data_dir, "GSM3453215_A549_4h_MOI02_NormCounts.npy"), group1.values)
np.save(os.path.join(data_dir, "GSM3453214_A549_4h_MOI20_NormCounts.npy"), group2.values)