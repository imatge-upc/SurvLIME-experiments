import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import copy

from sksurv.metrics import concordance_index_censored
import argparse
from typing import List, Callable
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# DL for survival
from pycox.models import CoxPH, DeepHitSingle

import torchtuples as tt
from pycox.evaluation import EvalSurv
import torch


from hyperparams import load_hyperparams
from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import Loader

# add seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def compute_boot_c_index(event_indicator, event_time, data, model, num_boot_rep):
    total_individuals = len(event_indicator)
    boot_c_index = []
    for j in range(num_boot_rep):
        idx = np.random.choice(np.arange(total_individuals), size=total_individuals, replace=True)
        event_indicator_boot = np.array([event_indicator[i] for i in idx])
        event_time_boot = np.array([event_time[i] for i in idx])
        data_boot = np.array([data[i] for i in idx])
        if sum(event_indicator_boot)>0:
            try:
                estimate_boot = model.predict_surv_df(data_boot)
                ev_boot = EvalSurv(estimate_boot, event_time_boot, event_indicator_boot, censor_surv='km')
                c_index_boot = ev_boot.concordance_td()
                boot_c_index.append(c_index_boot)
            except:
                pass
    boot_c_index = np.sort(np.array(boot_c_index))
    return np.mean(boot_c_index)

def train_model(args, model_name, data, label, dataset):
    """ Documentation train loop
    Arguments:
    args: Arguments from the parser
    data: Data to train
    label: Label to train
    """
    lr = args.lr
    # Transform data into numpy arrays
  # leave = [(col, None) for col in data[0].columns]
  # x_mapper = DataFrameMapper(leave)
  # x_train = x_mapper.fit_transform(data[0]).astype('float32')
  # x_val = x_mapper.transform(data[0]).astype('float32')
    label[0] = (np.asarray(label[0][0]), np.asarray(label[0][1]))
    label[2] = (np.asarray(label[2][0]), np.asarray(label[2][1]))
    y_test = label[2]

    in_features = data[0].shape[1]
    if model_name == 'deepsurv':
        out_features = 1
        y_train = label[0]
    elif model_name =='deephit':
        num_durations = 10
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(*label[0])
        label[0] = labtrans.transform(*label[0])
        out_features = labtrans.out_features
    
    # Model Hyperparms
    num_nodes = [args.num_nodes for i in range(args.num_layers)]

    batch_norm = args.batch_norm; dropout = args.dropout ;output_bias = args.output_bias
    batch_size = 16; epochs = args.epochs
    # Instantiate the model
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    
    if model_name == 'deepsurv':
        model = CoxPH(net, tt.optim.Adam(weight_decay=args.reg, lr=lr))
    elif model_name == 'deephit':
        model = DeepHitSingle(net, tt.optim.Adam(weight_decay=args.reg, lr=lr), duration_index=labtrans.cuts)
    

    x_train = data[0].astype('float32').to_numpy()
    x_val = data[0].astype('float32').to_numpy()
    x_test = data[2].astype('float32').to_numpy()

    callbacks = [tt.callbacks.EarlyStopping(patience=5)]
    # Train!
    log = model.fit(input=x_train, target=y_train, batch_size=batch_size,
                    epochs=epochs, callbacks=callbacks, verbose=False,
                    )
                        
    # if the output of the model is of length one
    if out_features == 1:
        model.compute_baseline_hazards()

    # Compute the C-index
    surv = model.predict_surv_df(x_train)
    ev = EvalSurv(surv, durations=y_train[1], events=y_train[0], censor_surv='km')
    c_index_train = ev.concordance_td()

    # Compte the C-index on the test set
    c_index_test = compute_boot_c_index(event_indicator=y_test[0], event_time=y_test[1], data=x_test, model=model,num_boot_rep= 100)
    
    print(f'Model: {model_name} -:- Dataset: {dataset}')
    print(f"C-index train: {c_index_train:.3f}")
    print(f"C-index test: {c_index_test:.3f}")
    return model

def load_hyperparms(args):
    names = ['num_layers', 'num_nodes', 'batch_norm', \
            'output_bias', 'dropout', 'reg', 'lr']
    values = [3, 16, True, True, 0.3, 0.0001, 0.01]

    for value, name in zip(values, names):
        setattr(args, name, value)
    return args

def models_and_datasets(args):
    if args.dataset == "all":
        datasets = ["udca", "lung", "veterans"]
    else:
        datasets = [args.dataset]
    if args.model == "all":
        models = ["deepsurv", "deephit"]
    else:
        models = []
        if 'deepsurv' in args.model:
            models.append('deepsurv')
        if 'deephit' in args.model:
            models.append('deephit')

    return datasets, models



def exp_real_datasets_dl(args_org):
    args = copy.deepcopy(args_org)
    save_dir = os.path.join(os.path.dirname(os.getcwd()), "computed_weights_csv", "exp_real_datasets")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets, models = models_and_datasets(args)

    for model_name in models:
        print("-"*50)
        for dataset in datasets:
            model_params  = load_hyperparams(model_name, dataset)
            for val, name in zip(model_params.values(), model_params.keys()):
                setattr(args, name, val)

            loader = Loader(dataset_name=dataset)
            x, events, times = loader.load_data()

            train, test = loader.preprocess_datasets(x, events, times, random_seed=0)
            # Prepare datasets
            events_train = [1 if x[0] else 0 for x in train[1]]
            times_train = [x[1] for x in train[1]]
            train[1] = (times_train, events_train)
            
            events_test = [1 if x[0] else 0 for x in test[1]]
            times_test = [x[1] for x in test[1]]
            test[1] = (times_test, events_test)

            data = [train[0], train[0], test[0]]
            label = [train[1], train[1], test[1]]
            model = train_model(args, model_name, data, label, dataset)
            print("-"*50)
            #import ipdb;ipdb.set_trace()
            times_to_fill = np.unique(train[1][0])
            times_to_fill.sort()
            #H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)
            def pred_funct(model,x):
                return model.predict_cumulative_hazards(x).T
            def pred_funct_surv(model,x):
                return model.predict_surv_df(x).T
            if model_name == 'deepsurv':
                predict_chf= partial(pred_funct, model)
                model_output_times = times_to_fill
                type_fn = 'cumulative'
            elif model_name == 'deephit':
                type_fn = 'survival'
                predict_chf= partial(pred_funct_surv, model)
                model_output_times = model.duration_index

            explainer = SurvLimeExplainer(
                    training_features=train[0].astype('float32').to_numpy(),
                    training_events=[True if x==1 else False for x in events_train],
                    training_times=times_train,
                    model_output_times=model_output_times,
                    random_state=10,
            )

           #computation_exp = explainer.montecarlo_explanation(data=test[0],
           #                                                   predict_fn=predict_chf,
           #                                                   type_fn=type_fn,
           #                                                   num_samples=1000,
           #                                                   num_repetitions=args.repetitions
           #                                                   )
           #                                     
           #file_name = f"{model_name}_exp_{dataset}_surv_weights.csv"
           #file_directory = os.path.join(save_dir, file_name)
           ## transform computation_exp to dataframe
           #computation_exp = pd.DataFrame(computation_exp, columns=test[0].columns)
           #computation_exp.to_csv(file_directory, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain SurvLIME results for a given dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="veterans",
        help="either veterans, lungs, udca or pbc, or all",
    )
    parser.add_argument(
        "--model", type=str, default="deepsurv", help="either deepsurv or deephit"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="How many times to repeat the experiment",
    )
    parser.add_argument(
        "--num_neighbours",
        type=int,
        default=1000,
        help="Number of neighbours to use for the explanation",
    )
    # add repetitions
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--output_bias', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_nodes', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--reg', type=float, default=0.001 )
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=18)

    args = parser.parse_args()
    exp_real_datasets_dl(args)
