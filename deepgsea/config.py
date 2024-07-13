import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchmetrics import Accuracy, AUROC, F1Score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from .interpret import print_concept_score
from .load_data import load_dataset
from .model import *

class Logger:

    def __init__(self, path, log=True):
        self.path = path
        self.log = log

    def __call__(self, content, **kwargs):
        print(content, **kwargs)
        if self.log:
            with open(self.path, 'a') as f:
                print(content, file=f, **kwargs)

class Config:
    
    def __init__(self, data, concept = "GO", model = "DeepGSEA", lr = 1e-4, max_epoch = 100, \
         train_batch_size = 64, test_batch_size = 64, test_step = 10, h_dim = 64, z_dim = 32, \
         n_layer_enc = 2, n_proto = 1, device = "cpu", seed = 0, fold = 0, \
         exp_str = None, eval = False, d_min = 1, lambda_1 = 0, lambda_2 = 0, lambda_3 = 0, \
         lambda_4 = 0, lambda_5 = 0, sigma = 0.1, weight_decay = 1e-2, ratio = 0.5, control = 0, \
         use_label = False, one_step = False, downsample = False):

        assert exp_str is not None

        self.seed = seed
        self.fold = fold
        self.device = device
        self.model_type = model
        self.data = data
        self.lr = lr
        self.max_epoch = max_epoch
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.test_step = test_step
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layer_enc = n_layer_enc
        self.n_proto = n_proto
        self.sigma = sigma
        self.weight_decay = weight_decay
        self.ratio = ratio
        self.control = control
        self.use_label = use_label
        self.one_step = one_step
        self.lambdas = {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "lambda_3": lambda_3,
            "lambda_4": lambda_4,
            "lambda_5": lambda_5,
        }

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        dataset = load_dataset(data, concept, fold=self.fold, seed=self.seed, control=self.control, downsample=downsample)
        self.train_set = dataset["train_set"]
        self.val_set = dataset["val_set"]
        self.test_set = dataset["test_set"]
        self.class_id = dataset["class_id"]
        self.c_id = dataset["c_id"]
        self.c_name = dataset["c_name"]
        self.gene_id = dataset["gene_id"]
        self.M = torch.from_numpy(dataset["c_mask"]).bool().to(self.device)
        self.n_concept = len(self.M)
        self.n_class = len(self.class_id)
        self.n_gene = len(self.gene_id)
        if self.n_class == 2:
            self.acc = Accuracy(task="binary", num_classes=2, average='macro').to(self.device)
            self.auroc = AUROC(task="binary", num_classes=2, average='macro').to(self.device)
            self.f1 = F1Score(task="binary", num_classes=2, average='macro').to(self.device)
        else:
            self.acc = Accuracy(task="multiclass", num_classes=self.n_class, average='macro').to(self.device)
            self.auroc = AUROC(task="multiclass", num_classes=self.n_class, average='macro').to(self.device)
            self.f1 = F1Score(task="multiclass", num_classes=self.n_class, average='macro').to(self.device)

        if self.train_set.label is None:
            self.use_label = False
        if self.use_label:
            self.idx2label = {i:l for i, l in enumerate(sorted(set(self.train_set.label)))}
            self.label2idx = {l:i for i, l in self.idx2label.items()}
            self.n_proto = len(self.idx2label)

        if self.model_type == "DeepGSEA":
            self.model = DeepGSEA(self.M, self.n_gene, self.h_dim, self.z_dim, self.n_layer_enc, \
                self.n_class, self.n_proto, self.sigma, d_min=d_min)
        else:
            raise ValueError("Model {:s} not supported".format(self.model_type))

        self.model.to(self.device)

        if data.split("_")[0] == "simul":
            self.checkpoint_dir = os.path.join("../checkpoint", data, control, self.model_type, concept, exp_str)
            self.log_dir = os.path.join("../log", data, control, self.model_type, concept, exp_str)
        else:
            self.checkpoint_dir = os.path.join("../checkpoint", data, self.model_type, concept, exp_str)
            self.log_dir = os.path.join("../log", data, self.model_type, concept, exp_str)

        if eval:
            self.logger = Logger(os.path.join(self.log_dir, "log.txt"), log=False)
        else:
            self.logger = Logger(os.path.join(self.log_dir, "log.txt"))
            self.tf_writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.collate_fn = self.my_collate
        self.ce_ = nn.CrossEntropyLoss(reduction="none")

        self.logger("*" * 40)
        self.logger("Dataset: {:s}".format(data))
        self.logger("Concept: {:s}".format(concept))
        self.logger("Model: {:s}".format(self.model_type))
        self.logger("Learning rate: {:f}".format(self.lr))
        self.logger("Max epoch: {:d}".format(self.max_epoch))
        self.logger("Batch size for training: {:d}".format(self.train_batch_size))
        self.logger("Batch size for testing: {:d}".format(self.test_batch_size))
        self.logger("Test step: {:d}".format(self.test_step))
        self.logger("Dimension of hidden layers: {:d}".format(self.h_dim))
        self.logger("Dimension of latent embeddings: {:d}".format(self.z_dim))
        self.logger("Number of hidden layers in concept encoder: {:d}".format(self.n_layer_enc))
        self.logger("Number of prototypes: {:d}".format(self.n_proto))
        self.logger("Sigma: {}".format(self.sigma))
        self.logger("Weigth decay: {}".format(self.weight_decay))
        self.logger("Concept training ratio: {}".format(self.ratio))
        self.logger("Lambdas: {}".format(self.lambdas))
        self.logger("Threshold for pairwise prototype distance: {}".format(d_min))
        self.logger("Use label: {}".format(self.use_label))
        self.logger("Device: {:s}".format(self.device))
        self.logger("Seed: {}".format(self.seed))
        self.logger("Fold: {}".format(self.fold))
        self.logger("*" * 40)
        self.logger("")

    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_loader = DataLoader(self.train_set, batch_size = self.train_batch_size, shuffle = True, collate_fn = self.collate_fn)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_epoch = 0
        best_metric = -1e9

        self.model.train()
        self.logger("Training Starts\n")
        
        save_name = "best_model.pt"
        lambda_2 = self.lambdas["lambda_2"]

        for epoch in range(self.max_epoch):
            if not self.one_step:
                if epoch == 0:
                    for name, param in self.model.named_parameters():
                        if name in ["importance", "bias"]:
                            param.requires_grad = False
                    save_name = "phase1_model.pt"
                    lambda_2 = 0
                elif epoch == round(self.max_epoch * self.ratio):
                    for name, param in self.model.named_parameters():
                        if name in ["importance", "bias"]:
                            param.requires_grad = True
                    save_name = "best_model.pt"
                    lambda_2 = self.lambdas["lambda_2"]

            self.logger("[Epoch {:d}]".format(epoch))
            train_loss = 0
            train_c_loss = 0
            start_time = time.time()
            y_truth = []
            y_logits = []
            
            for bat in train_loader:
                X = bat[0]
                y = bat[1]
                X = torch.tensor(X).float().to(self.device)
                y = torch.tensor(y).long().to(self.device)
                sample_size = len(X)
                optim.zero_grad()

                if self.use_label:
                    label = torch.tensor([self.label2idx[l] for l in bat[2]]).long().to(self.device)
                    logits, c_logits, losses = self.model(X, y, label)
                else:
                    logits, c_logits, losses = self.model(X, y)

                if self.model_type == "DeepGSEA":
                    c_loss = self.ce_(c_logits.reshape(-1, self.n_class), y.repeat_interleave(self.n_concept)).reshape(sample_size, self.n_concept)
                    p2p_loss = losses["p2p_loss"]
                    c2p_loss = losses["c2p_loss"]
                    p2c_loss = losses["p2c_loss"]
                    c_loss = c_loss.sum(1).mean()
                else:
                    c_loss = torch.tensor(0).to(self.device)
                    p2p_loss = torch.tensor(0).to(self.device)
                    c2p_loss = torch.tensor(0).to(self.device)
                    p2c_loss = torch.tensor(0).to(self.device)
                
                loss = self.ce_(logits, y).mean()
                loss = lambda_2 * loss + self.lambdas["lambda_1"] * c_loss + self.lambdas["lambda_3"] * p2p_loss \
                        + self.lambdas["lambda_4"] * c2p_loss + self.lambdas["lambda_5"] * p2c_loss
                
                loss.backward()
                optim.step()

                train_loss += loss.item() * sample_size
                train_c_loss += c_loss.item() * sample_size / self.n_concept
                y_truth.append(y)
                y_logits.append(torch.softmax(logits, dim=1))

            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            y_pred = y_logits.argmax(dim=1)

            train_acc = self.acc(y_pred, y_truth).cpu().item()
            if y_logits.shape[1] == 2:
                train_auc = self.auroc(y_logits[:,1], y_truth).cpu().item()
            else:
                train_auc = self.auroc(y_logits, y_truth).cpu().item()
            train_f1 = self.f1(y_pred, y_truth).cpu().item()

            self.logger("Time: {:.1f}s | Avg. Training Loss: {:.2f} | Avg. Training Concept Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. AUROC Score: {:.2f} | F1 Score: {:.2f}".format(time.time() - start_time, train_loss / len(self.train_set), train_c_loss / len(self.train_set), train_acc, train_auc, train_f1))

            self.tf_writer.add_scalar("loss/train", train_loss / len(self.train_set), epoch)
            self.tf_writer.add_scalar("accuracy/train", train_acc, epoch)
            self.tf_writer.add_scalar("auroc/train", train_auc, epoch)
            self.tf_writer.add_scalar("f1/train", train_f1, epoch)
            self.tf_writer.add_scalar("c_loss/train", train_c_loss / len(self.train_set), epoch)

            if (epoch + 1) % self.test_step == 0:
                self.model.eval()
                self.logger("[Evaluation on Validation Set]")
                start_time = time.time()

                val_loss, val_c_auc, val_acc, val_auc, val_f1 = self.evaluate(dataset="val")

                if self.model.importance.requires_grad:
                    curr_metric = (val_auc + val_f1) / 2 + val_c_auc + 0.1 * (train_auc + train_f1)
                else:
                    curr_metric = val_c_auc

                self.logger("Time: {:.1f}s | Avg. Validation Classfication Loss: {:.2f} | Avg. Validation Concept AUROC: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. AUROC Score: {:.2f} | F1 Score: {:.2f}".format(time.time() - start_time, val_loss, val_c_auc, val_acc, val_auc, val_f1))
                self.tf_writer.add_scalar("loss/val", val_loss, epoch)
                self.tf_writer.add_scalar("c_auc/val", val_c_auc, epoch)
                self.tf_writer.add_scalar("accuracy/val", val_acc, epoch)
                self.tf_writer.add_scalar("auroc/val", val_auc, epoch)
                self.tf_writer.add_scalar("f1/val", val_f1, epoch)
                
                if curr_metric > best_metric:
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, save_name))
                    best_metric = curr_metric
                    best_epoch = epoch
                    self.logger("Model Saved!")
                        
                self.model.train()

            self.logger("")
        
        self.logger("Best epoch: {:d}".format(best_epoch))
        self.logger("Training Ends\n")

        self.model.eval()
        self.logger("[Evaluation on Test Set]")
        start_time = time.time()
        test_loss, test_c_auc, test_acc, test_auc, test_f1 = self.evaluate(dataset="test", checkpoint_path=os.path.join(self.checkpoint_dir, "best_model.pt"))
        self.logger("Time: {:.1f}s | Avg. Test Classification Loss: {:.2f} | Avg. Test Concept AUROC: {:.2f} | Avg. Test Accuracy: {:.2f} | Avg. AUROC Score: {:.2f} | F1 Score: {:.2f}\n".format(time.time() - start_time, test_loss, test_c_auc, test_acc, test_auc, test_f1))
        if self.model_type == "DeepGSEA":
            self.interpret_dir = self.checkpoint_dir.replace("checkpoint", "interpret")
            if not os.path.exists(self.interpret_dir):
                os.makedirs(self.interpret_dir)
            print_concept_score(self)

    def my_collate(self, batch):
        X = [item[0].tolist() for item in batch]
        y = [item[1] for item in batch]
        if len(batch[0]) == 2:
            return X, y
        else:
            label = [item[2] for item in batch]
            return X, y, label
    
    def evaluate(self, dataset="test", checkpoint_path=None):
        if checkpoint_path is not None:
            print("Loading checkpoint from:", checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        if dataset == "train":
            loader = DataLoader(self.train_set, batch_size = self.test_batch_size, shuffle = False, collate_fn=self.collate_fn)
            n_sample = len(self.train_set)
        elif dataset == "val":
            loader = DataLoader(self.val_set, batch_size = self.test_batch_size, shuffle = False, collate_fn=self.collate_fn)
            n_sample = len(self.val_set)
        elif dataset == "test":
            loader = DataLoader(self.test_set, batch_size = self.test_batch_size, shuffle = False, collate_fn=self.collate_fn)
            n_sample = len(self.test_set)
        with torch.no_grad():
            
            y_truth = []
            y_logits = []
            c_y_logits = []

            for bat in loader:
                X = bat[0]
                y = bat[1]
                X = torch.tensor(X).float().to(self.device)
                y = torch.tensor(y).long().to(self.device)
                logits, c_logits, _ = self.model(X)
                y_truth.append(y)
                y_logits.append(logits)
                c_y_logits.append(c_logits)
            
            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            c_y_logits = torch.cat(c_y_logits) # (n_cell, n_concept, n_class)
            clf_loss = self.ce_(y_logits, y_truth).mean()
            y_logits = torch.softmax(y_logits, dim=-1)
            c_y_logits = torch.softmax(c_y_logits, dim=-1)
            y_pred = y_logits.argmax(dim=1)
            if self.model_type == "DeepGSEA":
                c_auc = []
                for idx in range(self.n_concept):
                    if c_y_logits.shape[-1] == 2:
                        c_auc.append(self.auroc(c_y_logits[:,idx,1], y_truth))
                    else:
                        c_auc.append(self.auroc(c_y_logits[:,idx], y_truth))
                c_auc = torch.stack(c_auc).topk(min(100, len(c_auc)))[0].mean().cpu().item()
            else:
                c_auc = 0

            acc = self.acc(y_pred, y_truth).cpu().item()
            if y_logits.shape[1] == 2:
                auc = self.auroc(y_logits[:,1], y_truth).cpu().item()
            else:
                auc = self.auroc(y_logits, y_truth).cpu().item()
            f1 = self.f1(y_pred, y_truth).cpu().item()

            return clf_loss, c_auc, acc, auc, f1