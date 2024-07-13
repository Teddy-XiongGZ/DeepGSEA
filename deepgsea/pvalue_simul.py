import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, combine_pvalues, false_discovery_control

def multiple_pvalue(fpath, fold=[0,1,2,3,4], pathway="complement ()"):
    idx = (pd.read_csv(os.path.join('_'.join([fpath, str(fold[0])]), "score", "Concept_Score.csv"), index_col=0).index == pathway).argmax()
    pvalues = []
    for i in fold:
        fpath_fold = os.path.join('_'.join([fpath, str(i)]), "score", "Concept_Score.csv")
        pvalues.append(pd.read_csv(fpath_fold, index_col=0).loc[pathway]['pvalue'])

    return combine_pvalues(pvalues, method='pearson').pvalue

if __name__ == "__main__":
    fold = [0,1,2,3,4]

    for exp in ["simul_1", "simul_2"]:
        pvalues_1 = []
        pvalues_2 = []
        control = ["0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1"]
        for c in control:
            print(">>>>>>>>>>>>{:s} - {:s}<<<<<<<<<<<<".format(exp, c))
            fpath = "../interpret/"+exp+"/"+c+"/DeepGSEA/HALLMARK/optimal"
            pvalues_1.append(multiple_pvalue(fpath, fold = fold, pathway = "complement ()"))
            pvalues_2.append(multiple_pvalue(fpath, fold = fold, pathway = "notch signaling ()"))
            print("pvalue for complement:", pvalues_1[-1])
            print("pvalue for notch signaling:", pvalues_2[-1])

        pd.DataFrame({"pval":pvalues_1, "pathway":["HALLMARK-COMPLEMENT"] * len(pvalues_1), "de":control, "method":["DeepGSEA"] * len(pvalues_1)}).to_csv("../interpret/"+exp+"/"+exp+"_1_ours.csv", index=False)
        pd.DataFrame({"pval":pvalues_2, "pathway":["HALLMARK-NOTCH-SIGNALING"] * len(pvalues_2), "de":control, "method":["DeepGSEA"] * len(pvalues_2)}).to_csv("../interpret/"+exp+"/"+exp+"_2_ours.csv", index=False)

    for exp in ["simul_3"]:
        pvalues_1 = []
        pvalues_2 = []
        control = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]
        for c in control:
            print(">>>>>>>>>>>>{:s} - {:s}<<<<<<<<<<<<".format(exp, c))
            fpath = "../interpret/"+exp+"/"+c+"/DeepGSEA/HALLMARK/optimal"
            pvalues_1.append(multiple_pvalue(fpath, fold = fold, pathway = "complement ()"))
            pvalues_2.append(multiple_pvalue(fpath, fold = fold, pathway = "notch signaling ()"))
            print("pvalue for complement:", pvalues_1[-1])
            print("pvalue for notch signaling:", pvalues_2[-1])

        pd.DataFrame({"pval":pvalues_1, "pathway":["HALLMARK-COMPLEMENT"] * len(pvalues_1), "de":control, "method":["DeepGSEA"] * len(pvalues_1)}).to_csv("../interpret/"+exp+"/"+exp+"_1_ours.csv", index=False)
        pd.DataFrame({"pval":pvalues_2, "pathway":["HALLMARK-NOTCH-SIGNALING"] * len(pvalues_2), "de":control, "method":["DeepGSEA"] * len(pvalues_2)}).to_csv("../interpret/"+exp+"/"+exp+"_2_ours.csv", index=False)

    for exp in ["simul_4"]:
        pvalues_1 = []
        pvalues_2 = []
        control = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        for c in control:
            print(">>>>>>>>>>>>{:s} - {:s}<<<<<<<<<<<<".format(exp, c))
            fpath = "../interpret/"+exp+"/"+c+"/DeepGSEA/HALLMARK/optimal"
            pvalues_1.append(multiple_pvalue(fpath, fold = fold, pathway = "complement ()"))
            pvalues_2.append(multiple_pvalue(fpath, fold = fold, pathway = "notch signaling ()"))
            print("pvalue for complement:", pvalues_1[-1])
            print("pvalue for notch signaling:", pvalues_2[-1])

        pd.DataFrame({"pval":pvalues_1, "pathway":["HALLMARK-COMPLEMENT"] * len(pvalues_1), "de":control, "method":["DeepGSEA"] * len(pvalues_1)}).to_csv("../interpret/"+exp+"/"+exp+"_1_ours.csv", index=False)
        pd.DataFrame({"pval":pvalues_2, "pathway":["HALLMARK-NOTCH-SIGNALING"] * len(pvalues_2), "de":control, "method":["DeepGSEA"] * len(pvalues_2)}).to_csv("../interpret/"+exp+"/"+exp+"_2_ours.csv", index=False)
