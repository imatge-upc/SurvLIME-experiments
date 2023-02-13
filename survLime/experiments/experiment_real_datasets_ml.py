import os
from typing import List, Callable
import warnings

warnings.filterwarnings("ignore")
import numpy as np
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

from hyperparams import load_hyperparams
from survlimepy.load_datasets import Loader
from survlimepy import SurvLimeExplainer

np.random.seed(42)


def compute_boot_c_index(
    event_indicator, event_time, estimate, num_boot_rep, concordance_fct, model="rsf"
):
    """Compute the c-index with bootstrapping"""
    total_individuals = len(event_indicator)
    boot_c_index = []
    for j in range(num_boot_rep):
        idx = np.random.choice(
            np.arange(total_individuals), size=total_individuals, replace=True
        )
        event_indicator_boot = np.array([event_indicator[i] for i in idx])
        event_time_boot = np.array([event_time[i] for i in idx])
        if sum(event_indicator_boot) > 0:
            if model == "xgb":
                label_boot = Surv.from_arrays(event_indicator_boot, event_time_boot)
                estimate_boot = estimate.iloc[idx]
                c_index_boot = concordance_fct(label_boot, estimate_boot)
            else:
                estimate_boot = np.array([estimate[i] for i in idx])
                c_index_boot = concordance_fct(
                    event_indicator_boot, event_time_boot, estimate_boot
                )[0]
            boot_c_index.append(c_index_boot)
    boot_c_index = np.sort(np.array(boot_c_index))
    return np.mean(boot_c_index)


def obtain_model(args, model_name):
    """Instantiate the model with the correct parameters"""

    if model_name == "cox":
        model = CoxPHSurvivalAnalysis(n_iter=100, alpha=0.0001)
        type_fn = "survival"
        predict_chf = partial(model.predict_survival_function, return_array=True)
    elif model_name == "rsf":
        model = RandomSurvivalForest(
            n_estimators=args.n_estimators,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            min_weight_fraction_leaf=args.min_weight_fraction_leaf,
            max_depth=args.max_depth,
            n_jobs=-1,
            random_state=42,
        )
        type_fn = "cumulative"
        predict_chf = partial(
            model.predict_cumulative_hazard_function, return_array=True
        )
    elif model_name == "xgb":
        DEFAULT_PARAMS = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": args.aft_loss_distribution,
            "aft_loss_distribution_scale": args.aft_loss_distribution_scale,
            "tree_method": "hist",
            "learning_rate": args.lr,
            "max_depth": args.max_depth,
            "alpha": args.alpha,
            "min_child_weight": args.min_child_weight,
        }
        model = XGBSEKaplanNeighbors(DEFAULT_PARAMS, n_neighbors=50)
        predict_chf = model.predict
        type_fn = "survival"
    else:
        raise AssertionError("Model not implemented")

    return model, predict_chf, type_fn


def obtain_c_index(model_name, dataset, model, test, train):
    """Obtain the c-index for the model"""
    preds = model.predict(test[0])
    preds_train = model.predict(train[0])

    event_indicator_test = [x[0] for x in test[1]]
    event_time_test = [x[1] for x in test[1]]

    event_indicator_train = [x[0] for x in train[1]]
    event_time_train = [x[1] for x in train[1]]
    if model_name == "xgb":
        c_index_test = compute_boot_c_index(
            event_indicator_test,
            event_time_test,
            preds,
            100,
            concordance_index,
            model=model_name,
        )
        c_index_train = concordance_index(train[1], preds_train)
    else:
        c_index_test = compute_boot_c_index(
            event_indicator_test,
            event_time_test,
            preds,
            100,
            concordance_index_censored,
        )
        c_index_train = concordance_index_censored(
            event_indicator_train, event_time_train, preds_train
        )[0]
    # wandb.log({"c_index train": c_index_train, "test c-index": c_index_test})
    print(f"Dataset: {dataset} and model  {model_name}")
    print(f"Train c-index: {c_index_train} and test c-index: {c_index_test}")


def models_and_datasets(args):
    if args.dataset == "all":
        datasets = ["veterans", "udca", "lung"]  # we are missing pbc
    else:
        datasets = [args.dataset]
    if args.model == "all":
        models = ["rsf", "xgb", "cox"]
    else:
        models = []
        if "cox" in args.model:
            models.append("cox")
        if "rsf" in args.model:
            models.append("rsf")
        if "xgb" in args.model:
            models.append("xgb")
    return datasets, models


def obtain_output_times(model_name, model):
    """Obtain the output times for the model. It is need for the explainer"""
    if model_name == "cox":
        model_output_times = model.event_times_
    elif model_name == "rsf":
        model_output_times = model.event_times_
    elif model_name == "xgb":
        model_output_times = model.time_bins
    else:
        raise AssertionError
    return model_output_times


def exp_real_datasets(args_org):
    """
    Experiments with real datasets
    These experiments correspond to the section 4.2 of the paper
    """
    import copy

    args = copy.deepcopy(args_org)

    save_dir = os.path.join(os.getcwd(), "computed_weights_csv", "exp_real_datasets")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    datasets, models = models_and_datasets(args)

    for model_name in models:
        print("-" * 50)
        for dataset in datasets:
            # Load and pre-process data
            loader = Loader(dataset_name=dataset)
            x, events, times = loader.load_data()
            if dataset=='lung':
                """
                We remove the column meal.cal as it has many missing values
                we remove any row with missing values of the LUNG dataset
                """
                x['event'] = events
                x['time'] = times
                x.drop('inst', axis=1, inplace=True)
                x.drop('meal.cal', axis=1, inplace=True)
                x.dropna(inplace=True)
                events = x.pop('event')
                times = x.pop('time')
            train, test = loader.preprocess_datasets(x, events, times, random_seed=0)

            # Obtain model and compute c-index
            if model_name != "cox":
                model_params = load_hyperparams(model_name, dataset)

                for val, name in zip(model_params.values(), model_params.keys()):
                    setattr(args, name, val)
            model, predict_chf, type_fn = obtain_model(args, model_name)
            model.fit(train[0], train[1])
            obtain_c_index(model_name, dataset, model, test, train)
                
            # Drop rows in the dataframe train[0] with 3 or more nan values


            model_output_times = obtain_output_times(model_name, model)
            events_train = [1 if x[0] else 0 for x in train[1]]
            times_train = [x[1] for x in train[1]]
            explainer = SurvLimeExplainer(
                training_features=train[0],
                training_events=[True if x == 1 else False for x in events_train],
                training_times=times_train,
                model_output_times=model_output_times,
                random_state=10,
            )
            computation_exp = explainer.montecarlo_explanation(
                data=test[0],
                predict_fn=predict_chf,
                type_fn=type_fn,
                num_samples=1000,
                num_repetitions=args.repetitions,
            )
            file_name = f"{model_name}_exp_{dataset}_surv_weights.csv"
            file_directory = os.path.join(save_dir, file_name)
            # transform computation_exp to dataframe
            computation_exp = pd.DataFrame(computation_exp, columns=test[0].columns)
            computation_exp.to_csv(file_directory, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain SurvLIME results for a given dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="veterans",
        help="either veterans, lung, udca or all",
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

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--reg", type=float, default=0.001, help="Regularization")

    # XGboost
    parser.add_argument("--aft_loss_distribution", type=str, default="normal")
    parser.add_argument("--aft_loss_distribution_scale", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=float, default=2)
    parser.add_argument(
        "--gamma", type=float, default=0.001
    )  # <- monomum loss reduction required to make further partition

    # Machine learning Arguments
    parser.add_argument("--n_iter", type=int, default=101)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_weight_fraction_leaf", type=float, default=0.00)
    parser.add_argument("--n_estimators", type=int, default=800)
    args = parser.parse_args()

    exp_real_datasets(args)
