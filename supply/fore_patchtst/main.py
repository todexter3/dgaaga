import argparse
import os
import torch
from exp.exp_kfold import Exp_Multiple_Regression_Fold
import random
import numpy as np
import os
from utils.str2bool import str2bool
import yaml
from types import SimpleNamespace

os.environ["KMP_AFFINITY"] = "noverbose"

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(8)



def main():
    if args.is_training:
        save_paths=args.save_path
        for args.tau_hat_init in [4.5,4.0, 3.0, 2.0, 1.0, 0.0]: 
            for args.learning_rate in [1e-5,3e-5,1e-4]:
                for args.batch_size in [256,128]:
                    for args.seq_len in [120]:    
                        for args.d_model in [128]:                 
                            args.d_ff = int(args.d_model * 2)
                            for args.patch_len in [16]:
                                args.stride = int(args.patch_len / 2)
                                
                                if args.data_type == 'daily':
                            
                                    args.pred_len = 1
                                set_seed(args.seed)
                                args.size = [args.seq_len, args.pred_len]
                                
                                train_des = f"task{args.task_name}_{args.model}_start_year{args.train_start_year}_end_year{args.train_end_year}_test_year{args.test_year}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                                model_des = f"dp{args.drop_ratio}_dmo{args.d_model}_dff{args.d_ff}"
                                patching_des = f'_pl{args.patch_len}_sr{args.stride}'
                                setting = train_des + model_des + patching_des
                                
                                args.save_path = os.path.join(save_paths, f'y{args.pred_task}/{args.model}_{setting}')
                                args.checkpoints = args.save_path
                                args.logs_dir = args.save_path + f'/logs'
                                if os.path.exists(args.save_path + '/pred.npy'):
                                    continue
                                if not os.path.exists(args.save_path):
                                    os.makedirs(args.save_path)
                                if not os.path.exists(args.logs_dir):
                                    os.makedirs(args.logs_dir)
                                print('args in experiment:')
                                print(args)
                                with open(f'{args.save_path}/_result_of_{args.task_name}.txt', 'a') as file:
                                    file.write('args in experiment:\n' + f'{args}\n\n')
                                
                                Exp = Exp_Multiple_Regression_Fold
                                exp = Exp(args)  # set experiments
                                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                                exp.train(setting)
                                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                                exp.test(setting)
                                torch.cuda.empty_cache()

    else:
        print('args in experiment:')
        print(args)
        if args.data_type == 'daily':
            args.pred_len = 1
        set_seed(args.seed)
        args.size = [args.seq_len, args.pred_len]
        # model = Model(args)
        train_des = f"task{args.task_name}_{args.model}_test_year{args.test_year}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
        model_des = f"dp{args.drop_ratio}_dmo{args.d_model}_dff{args.d_ff}"
        patching_des = f'_pl{args.patch_len}_sr{args.stride}'
        
        setting = train_des + model_des + patching_des
        args.save_path = os.path.join(args.save_path, f'y{args.pred_task}/{args.model}_{setting}')
        args.checkpoints = args.save_path
        args.logs_dir = args.save_path + f'/logs'
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(f'{args.save_path}/_result_of_{args.task_name}.txt', 'a') as file:
            file.write('args in experiment:\n' + f'{args}\n\n')
        print(f"🧩 Save path: {args.save_path}")
        print(f"🧩 Logs path: {args.logs_dir}")
        Exp = Exp_Multiple_Regression_Fold
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic config
    #parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    # task
    parser.add_argument('--task_name', type=str, default='multiple_regression',help='task name, options:multiple_regressio')
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    parser.add_argument('--pred_task', type=int, default=10, help='y5,y10,y20')

    # model
    parser.add_argument('--model', type=str, default='PatchTST_gc', help='model name, options: [PatchTST_gc]')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    # data loader
    parser.add_argument('--data_path', type=str, default='data/daily_2025.feather',help='data file, options: [ETT-small, electricity, exchange_rate, illness, traffic, weather]')
    parser.add_argument('--data_type', type=str, default='daily',help='date_type')                    
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_heiyi/', help='location of model checkpoints')
    parser.add_argument('--save_path', type=str, default='data/results/', help='train start year')
    parser.add_argument('--train_start_year', type=str, default='2010', help='train start year')
    parser.add_argument('--train_end_year', type=str, default='2019', help='train end year')
    parser.add_argument('--test_year', type=str, default= None, help='test year')
    parser.add_argument('--val_start_year', type=str, default='2014', help='vali start year')
    parser.add_argument('--ticker_type', type=int, default=3, help='ticker_type')

    # model define
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_channels', type=int,default=1)
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=16)

    # optimization
    parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--optim_type', type=str, default='Adam', help='select optimizer type, optional[SGD, Adam]')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay value')
    parser.add_argument('--loss', type=str, default='MSE_with_weak', help='loss function, optional[ MSE, MAE, CCC]')
    parser.add_argument('--lradj', type=str, default='not', help='adjust learning rate, optional:[type1, type2, not, cos, steplr]')
    parser.add_argument('--clip_value', type=float, default=0.5, help='clip grad')
    parser.add_argument('--pct_start', type=int, default=0.6)
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='Set a dropping ratio for feature_selection')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--tau_hat_init', type=float, default=0.0, help='tau_hat_init')
    parser.add_argument('--num_fold', type=int, default=5, help='')
    parser.add_argument('--grad_norm', type=bool, default=False, help='grad_norm')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # self.args.use_multi_gpu=1
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices =args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.dropout is None:
        args.dropout = args.drop_ratio

    # self.args.gpu = 2
    if args.test_year is None:
        args.test_year = str(int(args.train_end_year) + 1)
    
    args.val_start_year = str(int(args.test_year) - 1)
 
    args.save_path = os.path.join(args.save_path, f'{args.data_type}/')

    import subprocess, sys
    print("Python executable:", sys.executable)
    print("Torch version:", torch.__version__)
    try:
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("Current CUDA device:", torch.cuda.current_device())
            print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as e:
        print("CUDA check failed:")

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    main()
    