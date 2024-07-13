import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import tqdm
from goatools import obo_parser

comp_genes = ["ACTN2","ADAM9","ADRA2B","AKAP10","ANG","ANXA5","APOA4","APOBEC3F","APOBEC3G","APOC1","ATOX1","BRPF3","C1QA","C1QC","C1R","C1S","C2","C3","C4BPB","C9","CA2","CALM1","CALM3","CASP1","CASP10","CASP3","CASP4","CASP5","CASP7","CASP9","CBLB","CCL5","CD36","CD40LG","CD46","CD55","CD59","CDA","CDH13","CDK5R1","CEBPB","CFB","CFH","CLU","COL4A2","CP","CPM","CPQ","CR1","CR2","CSRP1","CTSB","CTSC","CTSD","CTSH","CTSL","CTSO","CTSS","CTSV","CXCL1","DGKG","DGKH","DOCK10","DOCK4","DOCK9","DPP4","DUSP5","DUSP6","DYRK2","EHD1","ERAP2","F10","F2","F3","F5","F7","F8","FCER1G","FCN1","FDX1","FN1","FYN","GATA3","GCA","GMFB","GNAI2","GNAI3","GNB2","GNB4","GNG2","GNGT2","GP1BA","GP9","GPD2","GRB2","GZMA","GZMB","GZMK","HNF4A","HPCAL4","HSPA1A","HSPA5","IL6","IRF1","IRF2","IRF7","ITGAM","ITIH1","JAK2","KCNIP2","KCNIP3","KIF2A","KLK1","KLKB1","KYNU","L3MBTL4","LAMP2","LAP3","LCK","LCP2","LGALS3","LGMN","LIPA","LRP1","LTA4H","LTF","LYN","MAFF","ME1","MMP12","MMP13","MMP14","MMP15","MMP8","MSRB1","MT3","NOTCH4","OLR1","PCLO","PCSK9","PDGFB","PDP1","PFN1","PHEX","PIK3CA","PIK3CG","PIK3R5","PIM1","PLA2G4A","PLA2G7","PLAT","PLAUR","PLEK","PLG","PLSCR1","PPP2CB","PPP4C","PRCP","PRDM4","PREP","PRKCD","PRSS3","PRSS36","PSEN1","PSMB9","RABIF","RAF1","RASGRP1","RBSN","RCE1","RHOG","RNF4","S100A12","S100A13","S100A9","SCG3","SERPINA1","SERPINB2","SERPINC1","SERPINE1","SERPING1","SH2B3","SIRT6","SPOCK2","SRC","STX4","TFPI2","TIMP1","TIMP2","TMPRSS6","TNFAIP3","USP14","USP15","USP16","USP8","VCPIP1","WAS","XPNPEP1","ZEB1","ZFPM2"]

def load_dataset(data, concept="pathway", data_dir=None, fold=0, seed=0, control='0', downsample=False):
    
    data = data.lower()

    if data == "glioblastoma":
        dataset = load_glioblastoma(data_dir = data_dir)    
    elif data == "influenza":
        dataset = load_influenza(data_dir = data_dir)
    elif data == "alzheimer":
        dataset = load_alzheimer(data_dir = data_dir)
    elif data.split("_")[0] == "simul":
        dataset = load_simul(data_dir = data_dir, data = data, control = control)
    else:
        raise ValueError("Data {:s} not supported".format(data))
    
    if downsample is not False:
        if downsample is True:
            downsample = 500
        for i in range(len(dataset["class_id"])):
            np.random.seed(0)
            x_idx = np.arange(len(dataset["groups"][i])).tolist()
            np.random.shuffle(x_idx)
            x_idx = x_idx[:downsample]
            dataset["groups"][i] = dataset["groups"][i].iloc[x_idx]
            if "labels" in dataset:
                dataset["labels"][i] = np.array(dataset["labels"][i])[x_idx].tolist()
    
    groups = dataset["groups"]
    class_id = dataset["class_id"]
    X = pd.concat(groups)
    y = [j for i in range(len(groups)) for j in [i] * len(groups[i])]
    X_y = X.copy()
    X_y['y'] = y
    if "labels" in dataset:
        labels = dataset["labels"]
        labels = [j for i in labels for j in i]
        X_y['label'] = labels
    else:
        labels = None

    if data.split("_")[0] == "simul":
        datasplit_dir = os.path.join("../data/datasplit", data + "_split", control)
    else:
        if downsample is not False:
            datasplit_dir = os.path.join("../data/datasplit", data + "_split", "downsample_" + str(downsample))
        else:
            datasplit_dir = os.path.join("../data/datasplit", data + "_split")

    # split the data into K-fold (K=5) and save the indices if there are no relevant local files
    if not os.path.exists(os.path.join(datasplit_dir, 'train%d.txt') % fold):
        if not os.path.exists(os.path.join(datasplit_dir)):
            os.makedirs(os.path.join(datasplit_dir))
        np.random.seed(0)
        x_len = len(X)
        # x_idx = np.arange(x_len).tolist()
        x_idx = X.index.tolist()
        K = 5
        while True:
            fold_split = []
            np.random.shuffle(x_idx)
            t_size = x_len // K + (x_len % K > 0) * 1
            for i in range(K):
                test_idx = x_idx[i * t_size : min(x_len, (i+1) * t_size)]
                train_idx = x_idx[:i * t_size] + x_idx[min(x_len, (i+1) * t_size):]
                if len(set(X_y["y"][test_idx])) + len(set(X_y["y"][train_idx])) != 2 * len(class_id):
                    break
                fold_split.append([test_idx, train_idx])
            if len(fold_split) == K:
                break
        for i in range(K):
            np.savetxt(os.path.join(datasplit_dir, 'test%d.txt') % i, fold_split[i][0], fmt="%s")
            np.savetxt(os.path.join(datasplit_dir, 'train%d.txt') % i, fold_split[i][1], fmt="%s")

    test_idx = np.loadtxt(os.path.join(datasplit_dir, 'test%d.txt') % fold, dtype=str).tolist()
    train_idx = np.loadtxt(os.path.join(datasplit_dir, 'train%d.txt') % fold, dtype="O").tolist()
    
    split_seed = seed
    train_idx_init = train_idx
    while True:
        train_idx, val_idx = train_test_split(train_idx_init, test_size=0.25, random_state=split_seed)
        if len(set(X_y["y"][val_idx])) + len(set(X_y["y"][train_idx])) == 2 * len(class_id):
            break
        split_seed += 1

    if concept == "ONE":
        if data == "alzheimer":
            genes = X.columns.tolist()
            gene2idx = {g: i for i, g in enumerate(genes)}
            database = json.load(open("../data/msigdb/c5.go.bp.v2023.1.Hs.json"))
            gc = "GOBP_GENERATION_OF_NEURONS"
            g_matched = []
            for g in database[gc]["geneSymbols"]:
                if g in genes:
                    g_matched.append(g)
            c_name = [' '.join(gc.split('_')[1:]).lower()]
            c_id = [database[gc]["exactSource"]]
            mask_idx = [gene2idx[g] for g in g_matched]
            c_mask = np.zeros((1, len(genes)))
            c_mask[:, mask_idx] = 1            
            gene_idx = c_mask.sum(0) > 0
            c_mask = c_mask[:, gene_idx]
            genes = X.columns[gene_idx].tolist()
        else:
            genes = comp_genes
            c_mask = np.ones((1, len(genes)))
            c_id = [""]
            c_name = ["HALLMARK_COMPLEMENT"]
            gene_idx = [True if g in genes else False for g in X.columns]
    elif concept == "full":
        genes = X.columns.tolist()
        c_mask = np.ones((1, len(genes)))
        c_id = [""]
        c_name = [""]
        gene_idx = [True] * len(genes)
    else:
        genes = X.columns.tolist()      
        c_mask, c_id, c_name, _, _ = create_concept_mask(genes, concept=concept)
        gene_idx = c_mask.sum(0) > 0
        c_mask = c_mask[:, gene_idx]
        genes = X.columns[gene_idx].tolist()
    
    train_set = OurDataset(X=X.loc[train_idx, gene_idx], y=X_y["y"][train_idx].tolist(), label=X_y["label"][train_idx].tolist() if labels is not None else None)
    val_set = OurDataset(X=X.loc[val_idx, gene_idx], y=X_y["y"][val_idx].tolist(), label=X_y["label"][val_idx].tolist() if labels is not None else None)
    test_set = OurDataset(X=X.loc[test_idx, gene_idx], y=X_y["y"][test_idx].tolist(), label=X_y["label"][test_idx].tolist() if labels is not None else None)

    return {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "c_mask": c_mask,
        "c_name": c_name,
        "c_id": c_id,
        "gene_id": genes,
        "class_id": class_id
    }

class OurDataset(Dataset):

    def __init__(self, X, y, label=None):
        self.X = X
        self.y = y
        self.label = label

    def __getitem__(self, i):
        if self.label is None:
            return self.X.iloc[i], self.y[i]
        else:
            return self.X.iloc[i], self.y[i], self.label[i]

    def __len__(self):
        return len(self.y)

def create_concept_mask(genes, concept="GO", min_genes=20, level=5):

    if concept == "GO":
        database = json.load(open("../data/msigdb/c5.go.bp.v2023.1.Hs.json"))
        GOdag = obo_parser.GODag(obo_file="../data/msigdb/go-basic.obo")
    elif concept == "HALLMARK":
        database = json.load(open("../data/msigdb/h.all.v2023.1.Hs.json"))
    else:
        database1 = json.load(open("../data/msigdb/h.all.v2023.1.Hs.json"))
        database2 = json.load(open("../data/msigdb/c2.cp.kegg.v2023.1.Hs.json"))
        database3 = json.load(open("../data/msigdb/c2.cp.reactome.v2023.1.Hs.json"))
        database = database1 | database2 | database3

    gene2idx = {g: i for i, g in enumerate(genes)}

    concept_name = []
    concept_id = []
    n_genes = []
    n_counts = []
    concept_mask = []

    if concept == "GO":
        print("Iterating through the GO database ...")
    elif concept == "HALLMARK":
        print("Iterating through the HALLMARK database ...")
    else:
        print("Iterating through the pathway databases ...")

    for gc in tqdm.tqdm(sorted(database.keys())):
        ngenes = len(database[gc]["geneSymbols"])
        if concept == "GO":
            if GOdag.get(database[gc]["exactSource"]):
                if GOdag.get(database[gc]["exactSource"]).level != level:
                    continue
            else:
                continue
        # if ngenes < min_genes:
        #     continue
        count = 0
        g_matched = []
        for g in database[gc]["geneSymbols"]:
            if g in genes:
                g_matched.append(g)
                count += 1
        if count < 1:
            continue
        concept_name.append(' '.join(gc.split('_')[1:]).lower())
        concept_id.append(database[gc]["exactSource"])
        n_genes.append(ngenes)
        n_counts.append(count)
        mask_idx = [gene2idx[g] for g in g_matched]
        g_mask = np.zeros(len(genes))
        g_mask[mask_idx] = 1
        concept_mask.append(g_mask)        

    concept_mask = np.stack(concept_mask)
    print("Number of concepts matched: {:d}".format(len(concept_mask)))

    return concept_mask, concept_id, concept_name, n_genes, n_counts

def preprocess_data(adata):
    # print("Preprocessing ...")
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    # print("Done!")
    return adata

def load_simul(data_dir = None, data = None, control = '0'):
    if data_dir is None:
        data_dir = "../data/simulation"
    
    data_dir = os.path.join(data_dir, data)

    group = sc.read_h5ad(os.path.join(data_dir, control + ".h5ad"))
    genes = group.var.index.tolist()
    group0 = pd.DataFrame(group.X[group.obs["group"] == 0].toarray(), index=group.obs.index[group.obs["group"] == 0].tolist(), columns=genes) 
    group1 = pd.DataFrame(group.X[group.obs["group"] == 1].toarray(), index=group.obs.index[group.obs["group"] == 1].tolist(), columns=genes) 

    return {
        "groups": [group0, group1],
        "class_id": ["group0", "group1"]
    }

def load_glioblastoma(data_dir = None, raw = False):
    
    if data_dir is None:
        data_dir = "../data/GSE132172"
    
    group = pd.read_csv(os.path.join(data_dir, "GSE132172_glio_chrom_instability_normalized_expression_matrix.tsv"), sep='\t')
    if raw:
        return group.T
    group0_label = [t for t in group.columns if "CB660" in t]
    group1_label = [t for t in group.columns if "GliNS2" in t]
    group0 = group[group0_label].T # (59, 21209)
    group1 = group[group1_label].T # (75, 21209)

    adata = sc.AnnData(np.concatenate([group0, group1]).astype(np.float32), var=pd.DataFrame(index=group0.columns.tolist()))

    sc.pp.scale(adata, max_value=10)

    genes = adata.var.index.tolist()
    group0 = pd.DataFrame(adata.X[:len(group0)], index=group0.index, columns=genes) # (59, )
    group1 = pd.DataFrame(adata.X[len(group0):], index=group1.index, columns=genes) # (75, )

    return {
        "groups": [group0, group1],
        "class_id": ["neural", "cancer"]
    }

def load_influenza(data_dir = None, raw = False):

    if data_dir is None:
        data_dir = "../data/GSE122031"

    group0_0 = pd.DataFrame(data = np.load(os.path.join(data_dir, "GSM3453216_A549_4h_Mock_NormCounts.npy")), 
                            index = open(os.path.join(data_dir, "barcodes_4h_mock.txt")).read().split('\n'),
                            columns = open(os.path.join(data_dir, "genes.txt")).read().split('\n'))
    group1_0 = pd.DataFrame(data = np.load(os.path.join(data_dir, "GSM3453215_A549_4h_MOI02_NormCounts.npy")), 
                            index = open(os.path.join(data_dir, "barcodes_4h_moi02.txt")).read().split('\n'),
                            columns = open(os.path.join(data_dir, "genes.txt")).read().split('\n'))
    group2_0 = pd.DataFrame(data = np.load(os.path.join(data_dir, "GSM3453214_A549_4h_MOI20_NormCounts.npy")), 
                            index = open(os.path.join(data_dir, "barcodes_4h_moi20.txt")).read().split('\n'),
                            columns = open(os.path.join(data_dir, "genes.txt")).read().split('\n'))

    if raw:
        return pd.concat([group0_0, group1_0, group2_0])
    adata = sc.AnnData(np.concatenate([group0_0, group1_0, group2_0]).astype(np.float32), var=pd.DataFrame(index=group0_0.columns.tolist()))
    adata = preprocess_data(adata)
    genes = adata.var.index.tolist()
    group0 = pd.DataFrame(adata.X[:len(group0_0)], index=group0_0.index.tolist(), columns=genes) # 
    group1 = pd.DataFrame(adata.X[len(group0_0):len(group0_0)+len(group1_0)], index=group1_0.index.tolist(), columns=genes) # 
    group2 = pd.DataFrame(adata.X[len(group0_0)+len(group1_0):], index=group2_0.index.tolist(), columns=genes) # 

    return {
        "groups": [group0, group1, group2],
        "class_id": ["Mock", "MOI0.2", "MOI2.0"],
    }

def load_alzheimer(data_dir = None):

    if data_dir is None:
        data_dir = "../data/SCP1375"

    genes = open(os.path.join(data_dir, "genes.txt")).read().strip().split("\n")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"), header=0, skiprows=[1])

    group = np.load(os.path.join(data_dir, "expression_matrix_normalized.npy")) # (72165, 2766)
    adata = sc.AnnData(group.astype(np.float32), var=pd.DataFrame(index=genes), obs={"batch": meta.batch.tolist()})

    barcodes = meta.NAME.astype(str)

    group0 = pd.DataFrame(adata.X[meta.label == "8months-control-replicate_1"], index = barcodes[meta.label == "8months-control-replicate_1"].tolist(), columns=genes) # (8506, 2766)
    group1 = pd.DataFrame(adata.X[meta.label == "8months-disease-replicate_1"], index = barcodes[meta.label == "8months-disease-replicate_1"].tolist(), columns=genes) # (8186, 2766)

    return {
        "groups": [group0, group1],
        "class_id": ["control", "disease"],
        "labels": [meta.top_level_cell_type[meta.label == "8months-control-replicate_1"].tolist(), \
                    meta.top_level_cell_type[meta.label == "8months-disease-replicate_1"].tolist()]
    }