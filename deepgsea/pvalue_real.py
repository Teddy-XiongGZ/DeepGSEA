import os
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
import argparse

def compute_pvalue(data):
    
    fold = [0,1,2,3,4]
    
    for database in ["pathway", "GO"]:
        dat = pd.DataFrame()
        for i in fold:
            fpath = os.path.join("../interpret", data, "DeepGSEA", database, "optimal_" + str(i), "score", "Concept_Score.csv")
            dat[str(i)] = pd.read_csv(fpath, index_col=0)["pvalue_adjust"]
        dat.index = pd.read_csv(fpath, index_col=0).index
        p_combined = []
        for i in range(len(dat)):
            p_combined.append(combine_pvalues(dat.iloc[i,:].tolist(), method='pearson').pvalue)
        dat["all"] = p_combined
        dat.to_csv(os.path.join("../interpret", data, "DeepGSEA", database, "optimal_pvalue.csv"))

if __name__ == "__main__":
    
    parser.add_argument("--data", default="glioblastoma")
    args = parser.parse_args()
    
    compute_pvalue(args.data)