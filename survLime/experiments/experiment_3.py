from typing import Union, List, Callable
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import wandb
import argparse
from typing import Union
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from xgbse.metrics import concordance_index
from xgbse._kaplan_neighbors import DEFAULT_PARAMS

from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import Loader


def obtain_model(args, model, dataset):
    if args.model == "cox":
        model = CoxPHSurvivalAnalysis(alpha=0.0001)
        type_fn = 'cumulative'
        predict_chf = partial(model.predict_survival_function, return_array=True)
    elif args.model == "rsf":
        if dataset=='udca':
            args.min_weight_fraction_leaf = 0.045
            args.min_samples_split = 4
            args.max_depth = 2
        else:
            args.min_weight_fraction_leaf = 0
            args.min_samples_split = 2
            args.max_depth = 3
        model = RandomSurvivalForest(n_estimators=args.n_estimators,
                                        max_features=args.max_features,
                                        min_samples_split=args.min_samples_split,
                                        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
                                        max_depth=args.max_depth,
                                        n_jobs=-1, random_state=42)
        type_fn = 'cumulative'
        predict_chf = partial(model.predict_survival_function, return_array=True)
    elif args.model == 'xgb':
        if dataset=='lung':
            alpha = 0.001; lr = 0.00507978667521773
            max_depth = 28; min_child_weight = 8
        else:
            alpha = args.alpha; lr =args.lr
            max_depth = args.max_depth; min_child_weight = args.min_child_weight
        DEFAULT_PARAMS = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': args.aft_loss_distribution,
            'aft_loss_distribution_scale': args.aft_loss_distribution_scale,
            'tree_method': 'hist', 
            'learning_rate': lr, 
            'max_depth': max_depth,
            'alpha': alpha,
            'min_child_weight': min_child_weight,
        }
        model = XGBSEKaplanNeighbors(DEFAULT_PARAMS, n_neighbors=50)
        predict_chf = model.predict
        type_fn = 'survival'
    else:
        raise AssertionError

    return model, predict_chf, type_fn



def obtain_c_index(args, dataset, model, test, train, x):
    # plot concordance index on test set
    preds = model.predict(test[0])
    preds_train = model.predict(train[0])
    if args.model == 'xgb':
        c_index = concordance_index(test[1], preds)
        c_index_train = concordance_index(train[1], preds_train)
    else:
        event_indicator = [x[0] for x in test[1]]
        event_time = [x[1] for x in test[1]]
        c_index = concordance_index_censored(event_indicator, event_time, preds)[0]
        # compute c_index_train
        event_indicator = [x[0] for x in train[1]]
        event_time = [x[1] for x in train[1]]
        c_index_train = concordance_index_censored(event_indicator, event_time, preds_train)[0]

    print(f'Dataset: {dataset} and model  {args.model}')
    print(f"Concordance index on test set: {c_index:.3f}")
    #wandb.log({"test c-index": c_index, "train c-index": c_index_train})


np.random.seed(42)
def main(args):
    #wandb.init(project="survlime", config=args)
    if args.dataset == "all":
        datasets = [ "udca", "lung", "pbc", "veterans"]
    else:
        datasets = [args.dataset]
    if args.model != "all":
        models = [args.model]
    else:
        models = ["cox", "rsf",'xgb']
    for model in models:
        print("-"*50)
        args.model = model
        for dataset in datasets:
            loader = Loader(dataset_name=dataset)
            x, events, times = loader.load_data()

            train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
            model, predict_chf, type_fn = obtain_model(args, model, dataset)

            model.fit(train[0], train[1])
            obtain_c_index(args, dataset, model, test, train, x)
            continue

            if args.model == "cox":
                model_output_times = model.event_times_
            elif args.model == "rsf":
                model_output_times = model.event_times_
            elif args.model == 'xgb':
                model_output_times = model.time_bins
            else:
                raise AssertionError
            events_train = [1 if x[0] else 0 for x in train[1]]
            times_train = [x[1] for x in train[1]]
            explainer = SurvLimeExplainer(
                    training_features=train[0],
                    training_events=[True if x==1 else False for x in events_train],
                    training_times=times_train,
                    model_output_times=model_output_times,
                    random_state=10,
            )
            for i in tqdm(range(args.repetitions)):

                computation_exp = compute_weights(explainer, 
                                                  test[0].astype('float32'),
                                                  predict_chf = predict_chf,
                                                  column_names = test[0].columns,
                                                  num_neighbors = args.num_neigh,
                                                type_fn = type_fn)
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
    for i, test_point in enumerate(tqdm(x_test.to_numpy())):
        b = explainer.explain_instance(
            test_point, predict_chf,
            verbose=False, num_samples=num_pat,
            type_fn = type_fn,
        )

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
        "--model", type=str, default="cox", help="bb model either cox or rsf"
    )
    parser.add_argument("--rs", type=int, default=0, help="Random seed for the splits")
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



    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--reg', type=float, default=0.001, help='Regularization')

    # XGboost
    parser.add_argument('--aft_loss_distribution', type=str, default='normal')
    parser.add_argument('--aft_loss_distribution_scale', type=float, default=1.0)
    parser.add_argument('--min_child_weight', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.001) # <- monomum loss reduction required to make further partition
    
    # Machine learning Arguments
    parser.add_argument('--n_iter', type=int, default=101)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0)
    parser.add_argument('--max_features', type=str, default='auto')
    args = parser.parse_args()

    main(args)

"""
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--reg', type=float, default=0.001, help='Regularization')

    # XGboost
    parser.add_argument('--aft_loss_distribution', type=str, default='normal')
    parser.add_argument('--aft_loss_distribution_scale', type=float, default=1.0)
    parser.add_argument('--min_child_weight', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.001) # <- monomum loss reduction required to make further partition
    
    # Machine learning Arguments
    parser.add_argument('--n_iter', type=int, default=101)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0)
"""
