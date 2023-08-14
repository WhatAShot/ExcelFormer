import os
import math
import time
import json
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from category_encoders import CatBoostEncoder

import optuna

from bin import ExcelFormer
from lib import Transformations, build_dataset, prepare_tensors, make_optimizer, DATA


DATASETS = [
    'analcatdata_supreme', 'isolet', 'cpu_act', 'visualizing_soil', 'yprop_4_1', 'gesture', 'churn', 'sulfur', 'bank-marketing', 'Brazilian_houses'
    'eye', 'MagicTelescope', 'Ailerons', 'pol', 'polv2', 'credit', 'california', 'house_sales', 'house', 'diamonds', 'helena', 'jannis', 'higgs-small',
    'road-safety', 'medical_charges', 'SGEMM_GPU_kernel_performance', 'covtype', 'nyc-taxi-green-dec-2016'
]


def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='config/ExcelFormer/mix_tuned')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=32, help='set a smaller early stop for fast hyper-tune or a larger for better performance, 32 is used in this paper')
    args = parser.parse_args()
    
    args.output = f'{args.output}/{args.dataset}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    # some basic model configuration
    cfg = {
        "model": {
            "prenormalization": True, # true or false, perform BETTER on a few datasets with no prenormalization 

            'kv_compression': None,
            'kv_compression_sharing': None,
            'token_bias': True
        },
        "training": {
            "max_epoch": 500,
            "optimizer": "adamw",
        }
    }
    
    return args, cfg


def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



"""args"""
device = torch.device('cuda')
args, cfg = get_training_args()
seed_everything(args.seed)

""" prepare Datasets and Dataloaders """
assert args.dataset in DATASETS
T_cache = False # save data preprocessing cache
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / args.dataset, transformation, T_cache)


if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
# convert categorical features to numerical features with CatBoostEncoder
if dataset.X_cat is not None:
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))), 
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    for k in ['train', 'val', 'test']:
        # 1: directly regard catgorical features as numerical
        dataset.X_num[k] = np.concatenate([enc.transform(dataset.X_cat[k]).astype(np.float32), dataset.X_num[k]], axis=1)

d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

X_cat = None # if use CatBoostEncoder then drop original categorical features

""" ORDER numerical features with MUTUAL INFORMATION """
mi_cache_dir = 'cache/mi'
if not os.path.isdir(mi_cache_dir):
    os.makedirs(mi_cache_dir)
mi_cache_file = f'{mi_cache_dir}/{args.dataset}.npy' # cache to save mutual information
if os.path.exists(mi_cache_file):
    mi_scores = np.load(mi_cache_file)
else:
    mi_func = mutual_info_regression if dataset.is_regression else mutual_info_classif
    mi_scores = mi_func(dataset.X_num['train'], dataset.y['train']) # calculate MI
    np.save(mi_cache_file, mi_scores)
mi_ranks = np.argsort(-mi_scores)
# reorder the feature with mutual information ranks
X_num = {k: v[:, mi_ranks] for k, v in X_num.items()}
# normalized mutual information for loss weight
sorted_mi_scores = torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum()).float().to(device)
""" END FEATURE REORDER """

# set batch size
batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256,
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'covtype': 1024
} # batch size settings for datasets in FT-Transformer(Borisov et al., 2021)
if args.dataset in batch_size_dict:
    batch_size = batch_size_dict[args.dataset]
    val_batch_size = 512
else:
    # batch size settings for datasets in (Grinsztajn et al., 2022)
    if dataset.n_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif dataset.n_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif dataset.n_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16

# update training config
cfg['training'].update({
    "batch_size": batch_size, 
    "eval_batch_size": val_batch_size, 
    "patience": args.early_stop
})

# data loaders
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}


"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

"""utils function"""
def apply_model(model, x_num, x_cat, mixup=True, beta=None, mtype:str=None):
    use_mixup = mixup and model.training
    return model(x_num, x_cat, mixup=use_mixup, beta=beta, mtype=mtype)
    
@torch.inference_mode()
def evaluate(model, parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            predictions[part].append(apply_model(model, x_num, x_cat, mixup=False))
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)


running_time = 0.
def train(model, optimizer, beta, mtype):
    """Training"""
    n_epochs = 500 # default max training epoch
    # warmup and lr scheduler
    warm_up = 10 # warm up epoch
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs - warm_up) # lr decay
    max_lr = cfg['training']['lr']

    best_score = -np.inf # record best validation score
    no_improvement = 0
    EARLY_STOP = args.early_stop

    global running_time
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        # warm up lr
        if warm_up > 0 and epoch <= warm_up:
            lr = max_lr * epoch / warm_up
            # print(f'warm up ({epoch}/{warm_up})')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        for iteration, batch in enumerate(train_loader):
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            optimizer.zero_grad()
            if mtype == 'none':
                loss = loss_fn(apply_model(model, x_num, x_cat, mixup=False), y)
            else:
                preds, feat_masks, shuffled_ids = apply_model(model, x_num, x_cat, mixup=True, beta=beta, mtype=mtype)
                if mtype == 'feat_mix':
                    lambdas = (sorted_mi_scores * feat_masks).sum(1) # bs
                    lambdas2 = 1 - lambdas
                elif mtype == 'hidden_mix':
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                elif args.mix_type == 'naive_mix':
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                if dataset.is_regression:
                    mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                    loss = loss_fn(preds, mix_y)
                else:
                    loss = lambdas * loss_fn(preds, y, reduction='none') + lambdas2 * loss_fn(preds, y[shuffled_ids], reduction='none')
                    loss = loss.mean()
            loss.backward()
            optimizer.step()
        
        scores = evaluate(model, ['val'])
        if dataset.is_binclass: # for binary classification use AUC metric
            val_score = scores['val']['roc_auc']
        else:
            val_score = scores['val']['score']
        if val_score > best_score:
            best_score = val_score
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement == EARLY_STOP:
            break
    running_time += time.time() - start
    return best_score


""" Prepare Model """
# datset specific params
n_num_features = dataset.n_num_features # drop some features
cardinalities = None

# model config and training config are fixed
kwargs = {
    'd_numerical': n_num_features,
    'd_out': d_out,
    'categories': cardinalities,
    **cfg['model'],
}
default_model_configs = {
    'ffn_dropout': 0., 'attention_dropout': 0.3, 'residual_dropout': 0.0,
    'n_layers': 3, 'n_heads': 32, 'd_token': 256,
    'init_scale': 0.01, # param for the Attenuated Initialization (keep fixed)
}
default_training_configs = {
    'lr': 1e-4,
    'weight_decay': 0.,
}
kwargs.update(default_model_configs)
cfg['training'].update(default_training_configs)
    

def objective(trial):
    # mix-tune hyper spaces
    beta = trial.suggest_float('beta', 0.1, 3.0) # Hyper range for mixup Beta Distribution
    mix_type = trial.suggest_categorical('mix_type', ['feat_mix', 'hidden_mix', 'none']) # mixup type range

    # build model
    model = ExcelFormer(**kwargs).to(device)

    # optimizer
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])    
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = make_optimizer(
        cfg['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        cfg['training']['lr'],
        cfg['training']['weight_decay'],
    )
    if torch.cuda.device_count() > 1:
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    
    best_val_score = train(model, optimizer, beta, mix_type)
    return best_val_score


cfg_file = f'{args.output}/cfg-tmp.json'
def save_per_iter(study, trial): # save searched config each iteration
    saved_model_cfg = {**cfg['model'], **default_model_configs} # model hyper is fixed
    saved_training_cfg = {**cfg['training'], **default_training_configs} # training hyper is fixed
    saved_mixup_cfg = {} # only tune mixup hyper
    for k in ['beta', 'mix_type']:
        saved_mixup_cfg[k] = study.best_trial.params.get(k)
    
    
    hyperparams = {
        'time': running_time,
        'eval_score': study.best_trial.value,
        'metric': 'rmse' if dataset.is_regression else 'roc_auc' if dataset.is_binclass else 'accuracy',
        'n_trial': study.best_trial.number,
        'dataset': args.dataset,
        'normalization': args.normalization,
        'model': saved_model_cfg,
        'training': saved_training_cfg,
        'mixup': saved_mixup_cfg,
    }
    with open(cfg_file, 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

iterations = 50 # search 50 iterations for mix tune
study = optuna.create_study(direction="maximize")
study.optimize(func=objective, n_trials=iterations, callbacks=[save_per_iter])

""" finish tuning """
# save the best config on validation set
final_cfg_file = f'{args.output}/cfg.json'
shutil.copyfile(cfg_file, final_cfg_file)