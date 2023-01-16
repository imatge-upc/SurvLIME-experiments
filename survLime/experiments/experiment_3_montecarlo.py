from typing import List, Callable
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import wandb
import argparse
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from xgbse import XGBSEKaplanNeighbors
from xgbse.metrics import concordance_index
from survlimepy.load_datasets import Loader
from survlimepy import SurvLimeExplainer

np.random.seed(42)

def compute_boot_c_index(event_indicator, event_time, estimate, num_boot_rep, concordance_fct, model='rsf'):
    total_individuals = len(event_indicator)
    boot_c_index = []
    for j in tqdm(range(num_boot_rep)):
        idx = np.random.choice(np.arange(total_individuals), size=total_individuals, replace=True)
        event_indicator_boot = np.array([event_indicator[i] for i in idx])
        event_time_boot = np.array([event_time[i] for i in idx])
        if sum(event_indicator_boot)>0:
            if model=='xgb':
                label_boot = Surv.from_arrays(event_indicator_boot, event_time_boot)
                estimate_boot = estimate.iloc[idx]
                c_index_boot = concordance_fct(label_boot, estimate_boot)
            else:
                estimate_boot = np.array([estimate[i] for i in idx])
                c_index_boot = concordance_fct(event_indicator_boot, event_time_boot, estimate_boot)[0]
            boot_c_index.append(c_index_boot)
    boot_c_index = np.sort(np.array(boot_c_index))
    return np.mean(boot_c_index)

def obtain_model(args, model, dataset):
    """ Instantiate the model with the correct parameters """

    if args.model == "cox":
        model = CoxPHSurvivalAnalysis(n_iter=args.n_iter, alpha=args.alpha)
        type_fn = 'survival'
        predict_chf = partial(model.predict_survival_function, return_array=True)
    elif args.model == "rsf":
       #if dataset=='udca':
       #    args.min_weight_fraction_leaf = 0.045
       #    args.min_samples_split = 4
       #    args.max_depth = 2
       #else:
       #    args.min_weight_fraction_leaf = 0
       #    args.min_samples_split = 2
       #    args.max_depth = 3
        model = RandomSurvivalForest(n_estimators=args.n_estimators,
                                        max_features=args.max_features,
                                        min_samples_split=args.min_samples_split,
                                        min_samples_leaf=args.min_samples_leaf,
                                        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
                                        max_depth=args.max_depth,
                                        n_jobs=-1, random_state=42)
        type_fn = 'cumulative'
        predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    elif args.model == 'xgb':
      # if dataset=='lung':
      #     alpha = 0.001; lr = 0.00507978667521773
      #     max_depth = 28; min_child_weight = 8
      # else:
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
        raise AssertionError("Model not implemented")

    return model, predict_chf, type_fn

def obtain_c_index(args, dataset, model, test, train):
    """ Obtain the c-index for the model """
    preds = model.predict(test[0])
    preds_train = model.predict(train[0])

    event_indicator_test = [x[0] for x in test[1]]
    event_time_test = [x[1] for x in test[1]]

    event_indicator_train = [x[0] for x in train[1]]
    event_time_train = [x[1] for x in train[1]]
    if args.model == 'xgb':
        c_index_test = compute_boot_c_index(event_indicator_test, event_time_test, preds,100,  concordance_index, model=args.model)
        c_index_train = concordance_index(train[1], preds_train)
    else:
        c_index_test = compute_boot_c_index(event_indicator_test, event_time_test, preds,100,  concordance_index_censored)
        c_index_train = concordance_index_censored(event_indicator_train, event_time_train, preds_train)[0]
    wandb.log({'c_index train': c_index_train, 'test c-index': c_index_test})
    print(f'Dataset: {dataset} and model  {args.model}')
    print(f'Train c-index: {c_index_train} and test c-index: {c_index_test}')

def models_and_datasets(args):
    if args.dataset == "all":
        datasets = [ "udca", "lung",  "veterans"] # we are missing pbc
    else:
        datasets = [args.dataset]
    if args.model != "all":
        models = [args.model]
    else:
        models = ["rsf",'xgb','cox'] # we are missing cox
    return datasets, models

def obtain_output_times(args, model):
    if args.model == "cox":
        model_output_times = model.event_times_
    elif args.model == "rsf":
        model_output_times = model.event_times_
    elif args.model == 'xgb':
        model_output_times = model.time_bins
    else:
        raise AssertionError
    return model_output_times


def load_hiperparams(args):
    """ Load the hyperparameters in case the exp is called from the main script"""
    names = ['lr', 'reg', 'aft_loss_distribution','aft_loss_distribution_scale', \
                        'min_child_weight', 'gamma','n_iter', 'alpha', 'tol', 'n_estimators', 'max_depth', 'min_samples_split' \
                        'min_weight_fraction_leaf', 'max_features']

    values = [0.0001, 0.001, 'normal', 1.0, 1, 0.001, \
                101, 0.0001, 1e-09, 100, 3, 2, 0, 'auto']

    for val, name in zip(values, names):
        setattr(args, name, val)
    return args

def exp_real_datasets(args, directly=False):
    if directly:
        args  = load_hiperparams(args)
    datasets, models = models_and_datasets(args)

    for model in models:
        print("-"*50)
        args.model = model
        for dataset in datasets:
            wandb.init(project="survlime", config=args)
            print(args)
            # Load and pre-process data
            loader = Loader(dataset_name=dataset)
            x, events, times = loader.load_data()
            train, test = loader.preprocess_datasets(x, events, times, random_seed=0)

            # Obtain model and compute c-index
            model, predict_chf, type_fn = obtain_model(args, model, dataset)
            model.fit(train[0], train[1])
            obtain_c_index(args, dataset, model, test, train)
            model_output_times = obtain_output_times(args, model)

            events_train = [1 if x[0] else 0 for x in train[1]]
            times_train = [x[1] for x in train[1]]
            explainer = SurvLimeExplainer(
                    training_features=train[0],
                    training_events=[True if x==1 else False for x in events_train],
                    training_times=times_train,
                    kernel_width=0.001,
                    model_output_times=model_output_times,
                    random_state=10,
            )
            computation_exp = explainer.montecarlo_explanation(data=test[0],
                                                               predict_fn=predict_chf,
                                                               type_fn=type_fn,
                                                               num_samples=1000,
                                                               num_repetitions=1
                                                               )
            # transform computation_exp to dataframe
            computation_exp = pd.DataFrame(computation_exp, columns=test[0].columns)
            save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp3_montecarlo/{args.model}_exp_{dataset}_surv_weights.csv"
            computation_exp.to_csv(save_path, index=False)

            computation_exp = explainer.montecarlo_explanation(data=test[0],
                                                               predict_fn=predict_chf,
                                                               type_fn=type_fn,
                                                               num_samples=1000,
                                                               num_repetitions=args.repetitions
                                                               )
            # transform computation_exp to dataframe
            computation_exp = pd.DataFrame(computation_exp, columns=test[0].columns)
            save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp3_montecarlo/{args.model}_exp_{dataset}_surv_weights.csv"
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
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--min_samples_leaf', type=int, default=19)
    parser.add_argument('--min_samples_split', type=int, default=7)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.05)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_features', type=str, default='auto')
    args = parser.parse_args()

    exp_real_datasets(args)

    """
    Hyperparameters rsf lung
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--min_samples_leaf', type=int, default=19)
    parser.add_argument('--min_samples_split', type=int, default=7)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.05)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_features', type=str, default='auto')
    """
    """
    parser.add_argument('--n_iter', type=int, default=101)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0)
    parser.add_argument('--max_features', type=str, default='auto')
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    """



    """
    Arguments for RSF for the LUNG dataset

    parser.add_argument('--n_iter', type=int, default=101)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--min_samples_split', type=int, default=4)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.025)
    parser.add_argument('--max_features', type=str, default='auto')
    parser.add_argument('--min_samples_leaf', type=int, default=6)
    """



    """
    Arguments used for UDCA for the paper
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
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=1)
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.05)
    parser.add_argument('--max_features', type=str, default='auto')
    parser.add_argument('--min_samples_leaf', type=int, default=17)
    """
