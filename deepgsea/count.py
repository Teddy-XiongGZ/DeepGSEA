import os
import re
import numpy as np

f = open("../log/count.txt", "w")

acc_total = []
auc_total = []
f1_total = []

walk_ls = sorted(os.walk("../log"))

for idx, i in enumerate(walk_ls):
    fdir = i[0]
    for fname in i[2]:
        if fname != "log.txt":
            continue
        last = open(os.path.join(fdir, fname)).read().strip().split("\n")[-1]
        if "Test Accuracy" not in last:
            continue
        acc = last.split(" | ")[-3].split(": ")[1] # a text
        auc = last.split(" | ")[-2].split(": ")[1] # a text
        f1 = last.split(" | ")[-1].split(": ")[1] # a text
        f.write(" | ".join([fdir, "acc: "+acc, "auc: "+auc, "f1: "+f1]) + "\n")
        acc_total.append(eval(acc))
        auc_total.append(eval(auc))
        f1_total.append(eval(f1))

    if idx == len(walk_ls)-1 or walk_ls[idx+1][0][:-1] != fdir[:-1]:
        if len(auc_total) > 0:
            f.write(" | ".join([fdir[:-1]+"_avg_mean", "acc: "+str(round(sum(acc_total) / len(acc_total), 2)), "auc: "+str(round(sum(auc_total) / len(auc_total), 2)), "f1: "+str(round(sum(f1_total) / len(f1_total), 2))]) + "\n")
            f.write(" | ".join([fdir[:-1]+"_avg_std", "acc: "+str(round(np.array(acc_total).std(), 2)), "auc: "+str(round(np.array(auc_total).std(), 2)), "f1: "+str(round(np.array(f1_total).std(), 2))]) + "\n")
            acc_total = []
            auc_total = []
            f1_total = []

f.close()