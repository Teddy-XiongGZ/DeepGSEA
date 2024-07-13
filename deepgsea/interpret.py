import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
import time
import torch
import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import mannwhitneyu,kruskal,combine_pvalues,false_discovery_control
from .load_data import *
import pdb

pal = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000', '#ffffff']
pal_proto = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000', '#ffffff']

def plot_pca_proto(X, label, label_set=None, prototype=None, prototype_label=None, save_path=None):

    assert prototype is not None

    start = time.time()
    if prototype is not None:
        n_proto = len(prototype)
    if label_set is None:
        label_set = set(label)

    trans = PCA(n_components=2)
    proto_pca = trans.fit_transform(prototype)
    X_pca = trans.transform(X)
    
    print("PCA Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))

    dot_color = {j:pal[i] for i, j in enumerate(sorted(label_set))}
    if prototype is not None:
        class_name = sorted(set(prototype_label))
        proto_color = {j:pal_proto[i] for i, j in enumerate(class_name)}

    plt.close()
    sns.set_theme(rc={'figure.figsize':(6,4.5)}, style="ticks")
    
    p = sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=label, s=15, palette=dot_color, alpha=0.8)

    p.set(xlabel="PCA1", ylabel="PCA2")
    p.set_xlabel(xlabel="PCA1", fontsize=14, weight='bold')
    p.set_ylabel(ylabel="PCA2", fontsize=14, weight='bold')

    if prototype is not None:
        for c in class_name:
            idx = [True if l == c else False for l in prototype_label]
            p.scatter(x=proto_pca[:,0][idx], y=proto_pca[:,1][idx], marker="*", s=200, color=proto_color[c], label=c, edgecolors='black')
        
    p.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size': 14})

    if save_path is not None:
        plt.savefig(save_path, bbox_extra_artists=(p,), bbox_inches='tight')

    plt.close()

def plot_umap(X, label, label_set=None, prototype=None, prototype_label=None, save_path=None, cmap=None):

    start = time.time()
    if prototype is not None:
        n_proto = len(prototype)
    if label_set is None:
        label_set = set(label)

    # umap-learn-based UMAP
    if prototype is not None:
        trans = umap.UMAP(init="pca", n_neighbors=min(30, X.shape[0]-1), random_state=42).fit(np.concatenate([X, prototype], axis=0))
        X_umap = trans.embedding_[:-n_proto]
        proto_umap = trans.embedding_[-n_proto:]
    else:
        trans = umap.UMAP(init="pca", n_neighbors=min(30, X.shape[0]-1), random_state=42).fit(X)
        X_umap = trans.embedding_
        
    print("UMAP Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))

    plt.clf()

    if label_set != "continuous":
        dot_color = {j:pal[i] for i, j in enumerate(sorted(label_set))}
        sns.set_theme(rc={'figure.figsize':(6,4.5)}, style="ticks")
    else:
        if cmap is not None:
            dot_color = sns.color_palette(cmap, as_cmap=True)
        else:
            dot_color = sns.color_palette("Blues", as_cmap=True)
        sns.set_theme(rc={'figure.figsize':(9,5.4)}, style="white")

    if prototype is not None:
        class_name = sorted(set(prototype_label))
        proto_color = {j:pal_proto[i] for i, j in enumerate(class_name)}
    

    fig, ax = plt.subplots()

    if label_set != "continuous":
        p = sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=label, s=15, palette=dot_color, alpha=0.8)
    else:
        p = sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=label, s=15, palette=dot_color, alpha=0.8, ax=ax)
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=dot_color, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm)
        p.legend_.remove()
    
    # p.set(xlabel="UMAP1", ylabel="UMAP2")
    p.set_xlabel(xlabel="UMAP1", fontsize=14, weight='bold')
    p.set_ylabel(ylabel="UMAP2", fontsize=14, weight='bold')

    if prototype is not None:
        p_labels = []
        for c in class_name:
            idx = [True if l == c else False for l in prototype_label]
            p_l = p.scatter(x=proto_umap[:,0][idx], y=proto_umap[:,1][idx], marker="*", s=200, color=proto_color[c], label=c, edgecolors='black')
            p_labels.append(p_l)

    if label_set != "continuous":
        p.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size': 14, 'weight':'bold'})
    else:
        p.legend(p_labels, class_name, prop={'size': 14, 'weight':'bold'})

    if save_path is not None:
        plt.savefig(save_path, bbox_extra_artists=(p,), bbox_inches='tight')

    plt.close()

def plot_tsne(X, label, label_set=None, prototype=None, prototype_label=None, save_path=None, cmap=None):

    start = time.time()
    if label_set is None:
        label_set = set(label)
    if prototype is not None:
        n_proto = len(prototype)

    tsne_transformer = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=min(30, len(X)-1), random_state=42)
    if prototype is not None:
        X_tsne = tsne_transformer.fit_transform(np.concatenate([X, prototype], axis=0))
        proto_tsne = X_tsne[-n_proto:]
        X_tsne = X_tsne[:-n_proto]
    else:
        X_tsne = tsne_transformer.fit_transform(X)

    print("t-SNE Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))

    plt.clf()

    if label_set != "continuous":
        dot_color = {j:pal[i] for i, j in enumerate(sorted(label_set))}
        sns.set_theme(rc={'figure.figsize':(6,4.5)}, style="ticks")
    else:
        if cmap is not None:
            dot_color = sns.color_palette(cmap, as_cmap=True)
        else:
            dot_color = sns.color_palette("Blues", as_cmap=True)
        sns.set_theme(rc={'figure.figsize':(9,5.4)}, style="white")

    if prototype is not None:
        class_name = sorted(set(prototype_label))
        proto_color = {j:pal_proto[i] for i, j in enumerate(class_name)}
    
    fig, ax = plt.subplots()

    if label_set != "continuous":
        p = sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=label, s=15, palette=dot_color, alpha=0.8)
    else:
        p = sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=label, s=15, palette=dot_color, alpha=0.8, ax=ax)
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=dot_color, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm)
        p.legend_.remove()
    
    # p.set(xlabel="tSNE1", ylabel="tSNE2")
    p.set_xlabel(xlabel="tSNE1", fontsize=14, weight='bold')
    p.set_ylabel(ylabel="tSNE2", fontsize=14, weight='bold')

    if prototype is not None:
        p_labels = []
        for c in class_name:
            idx = [True if l == c else False for l in prototype_label]
            p_l = p.scatter(x=proto_tsne[:,0][idx], y=proto_tsne[:,1][idx], marker="*", s=200, color=proto_color[c], label=c, edgecolors='black')
            p_labels.append(p_l)

    if label_set != "continuous":
        p.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size': 14})
    else:
        p.legend(p_labels, class_name)

    if save_path is not None:
        plt.savefig(save_path, bbox_extra_artists=(p,), bbox_inches='tight')
    
    plt.close()

def print_concept_score(config):

    with torch.no_grad():

        concept_id = [config.c_name[i] + " (" + config.c_id[i] + ")" for i in range(len(config.c_id))]
        save_dir = os.path.join(config.interpret_dir, "score")

        c_logits = []
        loader = DataLoader(config.test_set, batch_size = config.test_batch_size, shuffle = False, collate_fn=config.collate_fn)
        for bat in loader:
            X = bat[0]
            y = bat[1]
            X = torch.tensor(X).float().to(config.device)
            c_logits.append(config.model(X)[1])
        
        c_logits_soft = torch.softmax(torch.cat(c_logits), dim=-1) # (n_cell, n_concept, n_class)
        c_logits = torch.exp(torch.cat(c_logits))
        Y = torch.tensor(config.test_set.y).to(config.device)
        c_pred = c_logits.argmax(axis=-1)

        importance = config.model.compute_imp().squeeze(0).cpu().numpy()

        config.auroc = config.auroc.cpu()

        AUC = []
        print("Computing concept auROC scores.")
        for idx in tqdm.tqdm(range(c_logits_soft.shape[1])):
            if c_logits_soft.shape[2] == 2:
                AUC.append(config.auroc(c_logits_soft[:,idx][:,1], Y).cpu().item())
            else:
                AUC.append(config.auroc(c_logits_soft[:,idx], Y).cpu().item())
        AUC = np.array(AUC)

        c_logits = c_logits.cpu().numpy()
        Y = Y.cpu().numpy()
        c_pred = c_pred.cpu().numpy()

        pvalues = []
        print("Computing concept pvalues.")
        for idx in tqdm.tqdm(range(len(config.c_id))):
            p = combine_pvalues([mannwhitneyu(c_logits[Y == i,idx,i], c_logits[Y != i,idx,i], alternative="greater").pvalue for i in range(config.n_class)], method="fisher").pvalue
            pvalues.append(p)
        pvalues = np.array(pvalues)
        pvalues_adjust = false_discovery_control(pvalues,method='bh')

        subclass_idx = []
        subclass_name = []
        if config.test_set.label is None:
            subclass_name = config.class_id
            for i in range(len(config.class_id)):
                subclass_idx.append(Y == i)
        else:
            labels = config.test_set.label
            for l in sorted(set(labels)):
                subclass_name.append(l)
                subclass_idx.append(np.array(labels) == l)

        subclass_acc = []
        print("Computing subclass accuray.")
        for i in subclass_idx:
            acc = []
            for idx in tqdm.tqdm(range(c_logits.shape[1])):
                acc.append(accuracy_score(Y[i], c_pred[i,idx]))
            subclass_acc.append(np.array(acc))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pd.DataFrame(np.stack([AUC, pvalues, pvalues_adjust, importance] + subclass_acc).T, columns=["auROC", "pvalue", "pvalue_adjust", "importance"] + ["acc_" + i for i in subclass_name], index=concept_id).to_csv(os.path.join(save_dir, "Concept_Score.csv"))

def plot_phenotype_dist(config, concept_id=[], element_wise=False):

    with torch.no_grad():
        
        concept_auc = None

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                if element_wise:
                    concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist()
                else:
                    concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist() + imp_df.sort_values(by="pvalue", ascending=True).index[-3:].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
            concept_auc = imp_df.iloc[concept_idx]["auROC"].tolist()

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "phenotype")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        Z = []
        Y = []
        loader = DataLoader(config.test_set, batch_size = config.test_batch_size, shuffle = False, collate_fn=config.collate_fn)
        for bat in loader:
            X = bat[0]
            y = bat[1]
            X = torch.tensor(X).float().to(config.device)
            y = torch.tensor(y).long()            
            z = config.model.encode(X)
            Z.append(z.cpu()) # (n_cell, n_concept, z_dim)
            Y.append(y) # (n_cell)

        Z = torch.cat(Z).numpy()
        Y = [config.class_id[i] for i in torch.cat(Y)]
        Label = np.array(config.test_set.label)
    
        for i, c_idx in enumerate(concept_idx):
            if concept_auc is not None:
                file_name = concept_id[i] + "-" + concept_name[i] + " (" + str(concept_auc[i]) + ")"
            else:
                file_name = concept_id[i] + "-" + concept_name[i]

            if element_wise:
                
                prototype = config.model.prototypes[c_idx,:,:].cpu().numpy() # (n_class, n_proto, z_dim)
                idx2label = {i:l for i, l in enumerate(sorted(set(config.train_set.label)))}
                label2idx = {l:i for i, l in idx2label.items()}
                if prototype.shape[1] == len(idx2label):
                    for l in sorted(set(Label.tolist())):
                        try:
                            plot_umap(Z[:,c_idx][Label == l], np.array(Y)[Label == l].tolist(), label_set = set(config.class_id), prototype=prototype[:,label2idx[l]], prototype_label=config.class_id, save_path=os.path.join(save_dir, file_name + "_" + l + "_UMAP.png"))
                        except:
                            pass
                else:
                    prototype = prototype.reshape(-1, config.model.z_dim) # (n_class * n_proto, z_dim)
                    prototype_label = [j for i in config.class_id for j in [i] * config.model.n_proto] # (n_class * n_proto)  
                    for l in sorted(set(Label.tolist())):
                        try:
                            plot_umap(Z[:,c_idx][Label == l], np.array(Y)[Label == l].tolist(), label_set = set(config.class_id), prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_" + l + "_UMAP.png"))
                        except:
                            pass
            else:     
                prototype = config.model.prototypes[c_idx,:,:].reshape(-1, config.model.z_dim).cpu().numpy() # (n_class * n_proto, z_dim)
                prototype_label = [j for i in config.class_id for j in [i] * config.model.n_proto] # (n_class * n_proto)                       
                plot_umap(Z[:,c_idx], Y, label_set = set(config.class_id), prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_UMAP.png"))
                plot_pca_proto(Z[:,c_idx], Y, label_set = set(config.class_id), prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_PCA.png"))
            # plot_tsne(Z[:,c_idx], Y, label_set = set(config.class_id), prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_tSNE.png"))

def plot_label_dist(config, concept_id=[], element_wise=False):
    
    with torch.no_grad():
        
        concept_auc = None

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                if element_wise:
                    concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist()
                else:
                    concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist() + imp_df.sort_values(by="pvalue", ascending=True).index[-3:].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
            concept_auc = imp_df.iloc[concept_idx]["auROC"].tolist()

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "label")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        Z = []
        loader = DataLoader(config.test_set, batch_size = config.test_batch_size, shuffle = False, collate_fn=config.collate_fn)
        for bat in loader:
            X = bat[0]
            y = bat[1]
            X = torch.tensor(X).float().to(config.device)
            z = config.model.encode(X)
            Z.append(z.cpu()) # (n_cell, z_dim)
        Z = torch.cat(Z).numpy()
        Label = np.array(config.test_set.label)

        for i, c_idx in enumerate(concept_idx):
            if concept_auc is not None:
                file_name = concept_id[i] + "-" + concept_name[i] + " (" + str(concept_auc[i]) + ")"
            else:
                file_name = concept_id[i] + "-" + concept_name[i]
            prototype = config.model.prototypes[c_idx,:,:].reshape(-1, config.model.z_dim).cpu().numpy() # (n_class * n_proto, z_dim)
            prototype_label = [j for i in config.class_id for j in [i] * config.model.n_proto] # (n_class * n_proto)
            if element_wise:
                for l in sorted(set(Label.tolist())):
                    tmp = pd.DataFrame(np.concatenate([(Label == l).astype(int).reshape(-1,1), Z[:,c_idx]], axis=1)).sort_values(by=0)
                    plot_umap(tmp.iloc[:,1:].values, tmp.iloc[:,0].values, label_set = {0,1}, prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_" + l + "_UMAP.png"))
            else:
                plot_umap(Z[:,c_idx], Label, label_set = set(Label), prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_UMAP.png"))
                plot_pca_proto(Z[:,c_idx], Label, label_set = set(Label), prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_PCA.png"))

def plot_similarity_dist(config, concept_id=[]):
    
    with torch.no_grad():
        
        concept_auc = None

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist() + imp_df.sort_values(by="pvalue", ascending=True).index[-3:].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
            concept_auc = imp_df.iloc[concept_idx]["auROC"].tolist()

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "similarity")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        Z = []
        loader = DataLoader(config.test_set, batch_size = config.test_batch_size, shuffle = False, collate_fn=config.collate_fn)
        for bat in loader:
            X = bat[0]
            X = torch.tensor(X).float().to(config.device)
            z = config.model.encode(X)
            Z.append(z) # (n_cell, z_dim)
        Z = torch.cat(Z) # (n_cell, n_concept, z_dim)
        c2p_dist_sq = (torch.pow(Z[:, :, None, None, :] - config.model.prototypes[None, :, :, :, :], 2).sum(-1)) # (n_cell, n_concept, n_class, n_proto)
        c_logits = - (c2p_dist_sq / (2 * torch.exp(config.model.logvar)[None])).min(dim=-1)[0] + config.model.c_bias[None] # (n_cell, n_concept, n_class)
        c_logits = torch.exp(c_logits) # (n_cell, n_concept, n_class) # for similarity measurement
        
        Z = Z.cpu().numpy()
        c_logits = c_logits.cpu().numpy()

        for i, c_idx in enumerate(concept_idx):

            prototype = config.model.prototypes[c_idx,:,:].reshape(-1, config.model.z_dim).cpu().numpy() # (n_class * n_proto, z_dim)
            prototype_label = [j for i in config.class_id for j in [i] * config.model.n_proto] # (n_class * n_proto)

            predefined_cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]
            for j in range(len(config.class_id)):
                file_name = concept_id[i] + "-" + concept_name[i] + "_" + config.class_id[j]
                # predicts = c_logits[:,c_idx,j] * len(config.class_id) / (len(config.class_id) - 1) - c_logits[:,c_idx].sum(-1) / (len(config.class_id) - 1)
                predicts = c_logits[:, c_idx, j]
                plot_umap(Z[:,c_idx], predicts, label_set = "continuous", prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_UMAP.png"), cmap=predefined_cmaps[j])
                # plot_tsne(Z[:,c_idx], predicts, label_set = "continuous", prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_tSNE.png"), cmap=predefined_cmaps[j])

def plot_prediction_dist(config, concept_id=[]):
    
    with torch.no_grad():
        
        concept_auc = None

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist() + imp_df.sort_values(by="pvalue", ascending=True).index[-3:].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
            concept_auc = imp_df.iloc[concept_idx]["auROC"].tolist()

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "prediction")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        Z = []
        loader = DataLoader(config.test_set, batch_size = config.test_batch_size, shuffle = False, collate_fn=config.collate_fn)
        for bat in loader:
            X = bat[0]
            X = torch.tensor(X).float().to(config.device)
            z = config.model.encode(X)
            Z.append(z) # (n_cell, z_dim)
        Z = torch.cat(Z) # (n_cell, n_concept, z_dim)
        c2p_dist_sq = (torch.pow(Z[:, :, None, None, :] - config.model.prototypes[None, :, :, :, :], 2).sum(-1)) # (n_cell, n_concept, n_class, n_proto)
        c_logits = - (c2p_dist_sq / (2 * torch.exp(config.model.logvar)[None])).min(dim=-1)[0] + config.model.c_bias[None] # (n_cell, n_concept, n_class)
        c_logits = torch.softmax(c_logits, dim=-1) # (n_cell, n_concept, n_class) # for prediction measurement
        
        Z = Z.cpu().numpy()
        c_logits = c_logits.cpu().numpy()

        for i, c_idx in enumerate(concept_idx):

            prototype = config.model.prototypes[c_idx,:,:].reshape(-1, config.model.z_dim).cpu().numpy() # (n_class * n_proto, z_dim)
            prototype_label = [j for i in config.class_id for j in [i] * config.model.n_proto] # (n_class * n_proto)

            if len(config.class_id) == 2:
                file_name = concept_id[i] + "-" + concept_name[i]
                predicts = c_logits[:,c_idx,1]
                plot_umap(Z[:,c_idx], predicts, label_set = "continuous", prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_UMAP.png"), cmap="vlag")
            else:
                predefined_cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]
                for j in range(len(config.class_id)):
                    file_name = concept_id[i] + "-" + concept_name[i] + "_" + config.class_id[j]
                    predicts = c_logits[:, c_idx, j]
                    plot_umap(Z[:,c_idx], predicts, label_set = "continuous", prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_UMAP.png"), cmap=predefined_cmaps[j])
                    # plot_tsne(Z[:,c_idx], predicts, label_set = "continuous", prototype=prototype, prototype_label=prototype_label, save_path=os.path.join(save_dir, file_name + "_tSNE.png"), cmap=predefined_cmaps[j])

def plot_phenotype_origin_dist(config, concept_id=[]):

    with torch.no_grad():
        
        concept_auc = None

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist() + imp_df.sort_values(by="pvalue", ascending=True).index[-3:].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
            concept_auc = imp_df.iloc[concept_idx]["auROC"].tolist()

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "phenotype_origin")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        

        for i, c_idx in enumerate(concept_idx):

            X_all = (torch.tensor(config.test_set.X.values).float() * config.model.M[None, c_idx].cpu()).numpy()
            Y = [config.class_id[i] for i in config.test_set.y]

            if concept_auc is not None:
                file_name = concept_id[i] + "-" + concept_name[i] + " (" + str(concept_auc[i]) + ")"
            else:
                file_name = concept_id[i] + "-" + concept_name[i]
            plot_umap(X_all, Y, label_set = set(config.class_id), save_path=os.path.join(save_dir, file_name + "_UMAP.png"))
            # plot_tsne(X_all, Y, label_set = set(config.class_id), save_path=os.path.join(save_dir, file_name + "_tSNE.png"))
    
    del X_all

def plot_label_origin_dist(config, concept_id=[]):

    with torch.no_grad():
        
        concept_auc = None

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:3].tolist() + imp_df.sort_values(by="pvalue", ascending=True).index[-3:].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
            concept_auc = imp_df.iloc[concept_idx]["auROC"].tolist()

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "label_origin")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        

        for i, c_idx in enumerate(concept_idx):

            X_all = (torch.tensor(config.test_set.X.values).float() * config.model.M[None, c_idx].cpu()).numpy()
            Label = np.array(config.test_set.label)

            if concept_auc is not None:
                file_name = concept_id[i] + "-" + concept_name[i] + " (" + str(concept_auc[i]) + ")"
            else:
                file_name = concept_id[i] + "-" + concept_name[i]
            plot_umap(X_all, Label, label_set = set(Label), save_path=os.path.join(save_dir, file_name + "_UMAP.png"))
            # plot_tsne(X_all, Label, label_set = set(Label), save_path=os.path.join(save_dir, file_name + "_tSNE.png"))
    
    del X_all

def plot_heatmap(config, concept_id=[]):

    with torch.no_grad():

        if len(concept_id) == 0:
            if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
                imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)
                concept_id = imp_df.sort_values(by="pvalue", ascending=True).index[:1].tolist()
                concept_id = [i.split(" (")[-1][:-1] for i in concept_id]
            else:
                raise ValueError("Concept index list is empty.")

        concept_idx = [config.c_id.index(i) for i in concept_id]
        concept_name = [config.c_name[i] for i in concept_idx]
    
        if os.path.exists(os.path.join(config.interpret_dir, "score", "Concept_Score.csv")):
            imp_df = pd.read_csv(os.path.join(config.interpret_dir, "score", "Concept_Score.csv"), index_col=0)

        print("Concept:", concept_name)

        save_dir = os.path.join(config.interpret_dir, "heatmap")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        Z = []
        loader = DataLoader(config.test_set, batch_size = config.test_batch_size, shuffle = False, collate_fn=config.collate_fn)
        for bat in loader:
            X = bat[0]
            X = torch.tensor(X).float().to(config.device)
            z = config.model.encode(X)
            Z.append(z) # (n_cell, z_dim)
        Z = torch.cat(Z) # (n_cell, n_concept, z_dim)
        c2p_dist_sq = (torch.pow(Z[:, :, None, None, :] - config.model.prototypes[None, :, :, :, :], 2).sum(-1)) # (n_cell, n_concept, n_class, n_proto)
        c2p_dist_sq = (c2p_dist_sq / (2 * torch.exp(config.model.logvar)[None])) # (n_cell, n_concept, n_class, n_proto)

        Z = Z.cpu().numpy()
        c2p_dist_sq = c2p_dist_sq.cpu().numpy()
        y = np.array(config.test_set.y)

        if config.use_label:
            idx2label = {i:l for i, l in enumerate(sorted(set(config.train_set.label)))}
        else:
            idx2label = {i:str(i) for i in range(config.n_proto)}
            
        for i, c_idx in enumerate(concept_idx):
            plt.close()
            if config.data == "glioblastoma":
                X_all = load_glioblastoma(raw=True)
                X_all = X_all[config.gene_id].loc[config.test_set.X.index].values * config.model.M[None, c_idx].cpu().numpy()
            elif config.data == "influenza":
                X_all = load_influenza(raw=True)
                X_all = X_all[config.gene_id].loc[config.test_set.X.index].values * config.model.M[None, c_idx].cpu().numpy()
                X_all = np.log(X_all + 1)
            else:
                X_all = (torch.tensor(config.test_set.X.values).float() * config.model.M[None, c_idx].cpu()).numpy()
            var = pd.DataFrame(index=np.array(config.gene_id)[config.M[c_idx].cpu().numpy()].tolist())
            filtered_data = []
            group = []
            for j in range(config.n_class):
                for k in range(config.n_proto):
                    # T = 5
                    T = 10
                    sorted_idx = c2p_dist_sq[:, c_idx, j, k].argsort()
                    sorted_idx = [l for l in sorted_idx if y[l] == j]
                    filtered_data.append(X_all[sorted_idx[:T]][:,config.M[c_idx].cpu().numpy()])
                    if len(idx2label) == 1:
                        group += ([config.class_id[j]]*T)
                    else:
                        group += ([idx2label[k]+ '_' + config.class_id[j]]*T)
            filtered_data = np.concatenate(filtered_data)
            adata = sc.AnnData(filtered_data, var = var, obs={"group":group})
            rcParams['savefig.transparent'] = True
            sc.pl.heatmap(adata, adata.var.index.tolist(), groupby="group", figsize=(10, 4))
            plt.savefig(os.path.join(save_dir, concept_name[i]+".png"), bbox_inches='tight')

                    
if __name__ == "__main__":
    
    import argparse
    from config import Config

    parser = argparse.ArgumentParser()

    parser.add_argument("--interpret", type=str)
    parser.add_argument("--concept_id", nargs='+', type=str, default=[])
    parser.add_argument("--save_name", type=str, default="best_model.pt")
    parser.add_argument("--data", default="lupus")
    parser.add_argument("--concept", default="GO")
    parser.add_argument("--model", default="DeepGSEA")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--test_step", type=int, default=1)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--n_layer_enc", type=int, default=2)
    parser.add_argument("--n_proto", type=int, default=1, help="number of prototypes per class")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--exp_str", type=str, help="special string to identify an experiment")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--d_min", type=float, default=1.0)
    parser.add_argument("--lambda_1", type=float, default=0)
    parser.add_argument("--lambda_2", type=float, default=0)
    parser.add_argument("--lambda_3", type=float, default=0)
    parser.add_argument("--lambda_4", type=float, default=0)
    parser.add_argument("--lambda_5", type=float, default=0)
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--control", type=str, default='0')
    parser.add_argument("--use_label", action=argparse.BooleanOptionalAction)
    parser.add_argument("--one_step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--downsample", type=str, default='False')
    parser.add_argument("--element_wise", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    
    config = Config(
        data = args.data,
        concept = args.concept,
        model = args.model,
        lr = args.lr,
        max_epoch = args.max_epoch,
        train_batch_size = args.train_batch_size,
        test_batch_size = args.test_batch_size,
        test_step = args.test_step,
        h_dim = args.h_dim,
        z_dim = args.z_dim,
        n_layer_enc = args.n_layer_enc,
        n_proto = args.n_proto,
        device = args.device,
        seed = args.seed,
        fold = args.fold,
        exp_str = args.exp_str,
        eval = True,
        d_min = args.d_min,
        lambda_1 = args.lambda_1,
        lambda_2 = args.lambda_2,
        lambda_3 = args.lambda_3,
        lambda_4 = args.lambda_4,
        lambda_5 = args.lambda_5,
        sigma = args.sigma,
        weight_decay = args.weight_decay,
        ratio = args.ratio,
        control = args.control,
        use_label = False if args.use_label is None else True,
        one_step = False if args.one_step is None else True,
        downsample = eval(args.downsample)
    )
    
    print("Loading checkpoint from:", os.path.join(config.checkpoint_dir, args.save_name))
    config.model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, args.save_name), map_location=config.device))
    config.model.eval()

    config.interpret_dir = config.checkpoint_dir.replace("checkpoint", "interpret")
    if not os.path.exists(config.interpret_dir):
        os.makedirs(config.interpret_dir)
    
    interpret = args.interpret
    concept_id = args.concept_id
    element_wise = False if args.element_wise is None else True

    if interpret == "score":
        print_concept_score(config)
    elif interpret == "phenotype":
        plot_phenotype_dist(config, concept_id=concept_id, element_wise=element_wise)
    elif interpret == "label":
        plot_label_dist(config, concept_id=concept_id, element_wise=element_wise)
    elif interpret == "similarity":
        plot_similarity_dist(config, concept_id=concept_id)
    elif interpret == "prediction":
        plot_prediction_dist(config, concept_id=concept_id)
    elif interpret == "phenotype_origin":
        plot_phenotype_origin_dist(config, concept_id=concept_id)
    elif interpret == "label_origin":
        plot_label_origin_dist(config, concept_id=concept_id)
    elif interpret == "heatmap":
        plot_heatmap(config, concept_id=concept_id)
    else:
        raise ValueError("Interpretation \"{:s}\" not supported".format(interpret))