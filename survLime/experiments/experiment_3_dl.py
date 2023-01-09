import wandb
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

import argparse
from typing import Union, List, Callable
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# DL for survival
import pycox
from pycox.models import CoxPH, DeepHitSingle

from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.evaluation import EvalSurv
import torch

from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import Loader

# add seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_model(args, data, label, dataset):
    """ Documentation train loop
    Arguments:
    args: Arguments from the parser
    data: Data to train
    label: Label to train
    """
    if args.model == 'deephit':
        if dataset == 'pbc':
            lr = 0.001
        else:
            lr = args.lr
    print(lr)
    # Transform data into numpy arrays
  # leave = [(col, None) for col in data[0].columns]
  # x_mapper = DataFrameMapper(leave)
  # x_train = x_mapper.fit_transform(data[0]).astype('float32')
  # x_val = x_mapper.transform(data[1]).astype('float32')
    label[0] = (np.asarray(label[0][1]), np.asarray(label[0][0]))
    label[1] = (np.asarray(label[1][1]), np.asarray(label[1][0]))
    label[2] = (np.asarray(label[2][1]), np.asarray(label[2][0]))

    # Model Hyperparms
    num_nodes = [args.num_nodes for i in range(args.num_layers)]

    batch_norm = args.batch_norm; dropout = args.dropout ;output_bias = args.output_bias
    batch_size = 16; epochs = 1000
    callbacks = [tt.callbacks.EarlyStopping()]

    in_features = data[0].shape[1]
    if args.model == 'deepsurv':
        out_features = 1
        y_train = label[0]
    elif args.model =='deephit':
        num_durations = 20
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(*label[0])
        label[1] = labtrans.transform(*label[1])
        out_features = labtrans.out_features
    
    # Instantiate the model
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    
    if args.model == 'deepsurv':
        model = CoxPH(net, tt.optim.Adam(weight_decay=args.reg))
    elif args.model == 'deephit':
        model = DeepHitSingle(net, tt.optim.Adam(weight_decay=args.reg), duration_index=labtrans.cuts)
    callbacks = [tt.callbacks.EarlyStopping(patience=5)]

    
    model.optimizer.set_lr(lr)

    x_train = data[0].astype('float32').to_numpy()
    x_val = data[1].astype('float32').to_numpy()
    x_test = data[2].astype('float32').to_numpy()

    #val = x_val, label[1]

    # Train!
    log = model.fit(input=x_train, target=y_train, batch_size=batch_size,
                    epochs=epochs, callbacks=callbacks, verbose=False,
                    val_data=(x_val, label[1]))
                        
    # if the output of the model is of length one
    if out_features == 1:
        model.compute_baseline_hazards()

    try:
        # Compute the C-index
        surv = model.predict_surv_df(x_train)
        ev = EvalSurv(surv, y_train[0], y_train[1], censor_surv='km')
        c_index = ev.concordance_td('antolini')

        # Compte the C-index on the test set
        y_test = label[2]
        surv_test = model.predict_surv_df(x_test)
        #surv = surv.T
        ev_test = EvalSurv(surv_test, y_test[0], y_test[1], censor_surv='km')
        c_index_test = ev_test.concordance_td('antolini')
    except:
        # print traceback
        import traceback
        traceback.print_exc()
        #c_index = 0
        c_index_test = 0
        print('Peto')
    #wandb.log({'c_index': c_index, 'test c-index': c_index_test})
    print(f'Model: {args.model} -:- Dataset: {dataset}')
    print(f"C-index train: {c_index:.3f}")
    print(f"C-index test: {c_index_test:.3f}")

    return model

def main(args):
    #wandb.init(project="survlime", config=args)
    if args.dataset == "all":
        datasets = ["udca", "lung", "pbc", "veterans"]
    else:
        datasets = [args.dataset]
    if args.model != "all":
        models = [args.model]
    else:
        models = ["deepsurv", "deephit"]
    for model in models:
        print("-"*50)
        args.model = model
        for dataset in datasets:
            loader = Loader(dataset_name=dataset)
            x, events, times = loader.load_data()

            train, val, test = loader.preprocess_datasets(x, events, times, random_seed=0)
            for i in tqdm(range(args.repetitions)):
                # Prepare datasets
                events_train = [1 if x[0] else 0 for x in train[1]]
                times_train = [x[1] for x in train[1]]
                train[1] = (events_train, times_train)
                
                events_val = [1 if x[0] else 0 for x in val[1]]
                times_val = [x[1] for x in val[1]]
                val[1] = (events_val, times_val)

                events_test = [1 if x[0] else 0 for x in test[1]]
                times_test = [x[1] for x in test[1]]
                test[1] = (events_test, times_test)

                data = [train[0], val[0], test[0]]
                label = [train[1], val[1], test[1]]
                model = train_model(args, data, label, dataset)

                continue
                times_to_fill = list(set([x[1] for x in train[1]]))
                times_to_fill.sort()
                #H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)
                def pred_funct(model,x):
                    return model.predict_surv_df(x).T
                if args.model == 'deepsurv':
                    predict_chf = model.predict_surv_df
                    predict_chf= partial(pred_funct, model)
                   #chosen_H0 = model.baseline_hazards_.to_numpy()
                   #chosen_H0 = np.reshape(chosen_H0, (chosen_H0.shape[0],1))
                    #print(chosen_H0.shape)
                    model_output_times = times_to_fill
                    type_fn = 'survival'
                else:
                    type_fn = 'survival'
                    predict_chf = model.predict_surv_df
                    model_output_times = model.duration_index

                explainer = SurvLimeExplainer(
                        training_features=train[0].astype('float32').to_numpy(),
                        training_events=[True if x==1 else False for x in events_train],
                        training_times=times_train,
                        model_output_times=model_output_times,
                        random_state=10,
                )

                computation_exp = compute_weights(explainer, 
                                                  test[0].astype('float32'),
                                                  predict_chf = predict_chf,
                                                  column_names = test[0].columns,
                                                  num_neighbors = args.num_neighbours,
                                                type_fn = type_fn)
                                                 
              # computation_exp = compute_weights(explainer, test[0], model, num_neighbors=args.num_neigh)
                save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp3/{args.model}_exp_{dataset}_surv_weights_na_rand_seed_{i}.csv"
                computation_exp.to_csv(save_path, index=False)


def compute_weights(
    explainer: SurvLimeExplainer,
    x_test: np.ndarray,
    column_names: List[str],
    predict_chf: Callable,
    num_neighbors: int = 1000,
    type_fn: str = 'cumulative'
):
    compt_weights = []
    num_pat = num_neighbors
    #predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    for i, test_point in enumerate(tqdm(x_test.to_numpy())):
        b = explainer.explain_instance(
            test_point, predict_chf,
            verbose=False, num_samples=num_pat,
            type_fn = type_fn
        )

#           #print traceback
#           import traceback
#           traceback.print_exc()
#           print(f"Error in the computation of the weights in patient {i}")
#           b = [None] * len(test_point)
        compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=column_names)


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
        "--num_neigh",
        type=int,
        default=1000,
        help="Number of neighbours to use for the explanation",
    )

    parser.add_argument('--num_neighbours', type=int, default=1000, help='Number of neighbours to use')

    # add repetitions
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--output_bias', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--reg', type=float, default=0.0001 )
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    print(args)
    main(args)
# These work sufficiently good for DS [udca, lung, veterans] and DH [udca, pbc]
# PBC deephit
   #parser.add_argument('--num_layers', type=int, default=2)
   #parser.add_argument('--num_nodes', type=int, default=32)
   #parser.add_argument('--batch_norm', type=bool, default=False)
   #parser.add_argument('--dropout', type=float, default=0.2)
   #parser.add_argument('--output_bias', type=bool, default=False)
   #parser.add_argument('--reg', type=float, default=0.0001 )
   #parser.add_argument('--lr', type=float, default=0.001)


   #parser.add_argument('--num_layers', type=int, default=1)
   #parser.add_argument('--num_nodes', type=int, default=32)
   #parser.add_argument('--batch_norm', type=bool, default=False)
   #parser.add_argument('--dropout', type=float, default=0.3)
   #parser.add_argument('--output_bias', type=bool, default=False)
   #parser.add_argument('--reg', type=float, default=0.0001)
   #parser.add_argument('--lr', type=float, default=0.2)

# For DeepSurv and PBC

   #parser.add_argument('--num_layers', type=int, default=2)
   #parser.add_argument('--num_nodes', type=int, default=8)
   #parser.add_argument('--batch_norm', type=bool, default=False)
   #parser.add_argument('--dropout', type=float, default=0.2)
   #parser.add_argument('--output_bias', type=bool, default=False)
   #parser.add_argument('--reg', type=float, default=0.0001)
   #parser.add_argument('--lr', type=float, default=0.01)

   #parser.add_argument('--num_layers', type=int, default=2)
   #parser.add_argument('--num_nodes', type=int, default=16)
   #parser.add_argument('--batch_norm', type=bool, default=False)
   #parser.add_argument('--dropout', type=float, default=0.2)
   #parser.add_argument('--output_bias', type=bool, default=False)
   #parser.add_argument('--reg', type=float, default=0.0001)
   #parser.add_argument('--lr', type=float, default=0.01)
