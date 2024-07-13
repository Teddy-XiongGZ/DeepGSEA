import os
import time
import torch
import argparse
from .config import Config
import pdb

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="lupus")
    parser.add_argument("--concept", default="pathway")
    parser.add_argument("--model", default="DeepGSEA")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
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
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--control", type=str, default='0')
    parser.add_argument("--use_label", action=argparse.BooleanOptionalAction)
    parser.add_argument("--one_step", action=argparse.BooleanOptionalAction)

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
        one_step = False if args.one_step is None else True
    )
    
    config.model.eval()
    config.logger("[Evaluation on Test Set]")
    start_time = time.time()
    test_loss, test_c_auc, test_acc, test_auc, test_f1 = config.evaluate(dataset="test", checkpoint_path=os.path.join(config.checkpoint_dir, "best_model.pt"))
    config.logger("Time: {:.1f}s | Avg. Test Classification Loss: {:.2f} | Avg. Test Concept AUROC: {:.2f} | Avg. Test Accuracy: {:.2f} | Avg. AUROC Score: {:.2f} | F1 Score: {:.2f}\n".format(time.time() - start_time, test_loss, test_c_auc, test_acc, test_auc, test_f1))