import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

import argparse
from typing import Union
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from survLime import survlime_explainer
from survLime.datasets.load_datasets import Loader


def main(args):
    if args.dataset == "all":
        datasets = ["veterans", "udca", "lung", "pbc"]
    else:
        datasets = [args.dataset]
    if args.model != "all":
        models = [args.model]
    else:
        models = ["cox", "rsf"]
    for model in models:
        args.model = model
        for dataset in datasets:
            for i in tqdm(range(args.repetitions)):
                loader = Loader(dataset_name=dataset)
                x, events, times = loader.load_data()

                train, _, test = loader.preprocess_datasets(x, events, times, random_seed=i)

                if args.model == "cox":
                    model = CoxPHSurvivalAnalysis(alpha=0.0001)
                elif args.model == "rsf":
                    model = RandomSurvivalForest()
                else:
                    raise AssertionError

                model.fit(train[0], train[1])

                times_to_fill = list(set([x[1] for x in train[1]]))
                times_to_fill.sort()
                #H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)

                explainer = survlime_explainer.SurvLimeExplainer(
                    train[0], train[1], model_output_times=model.event_times_
                )

                computation_exp = compute_weights(explainer, test[0], model, num_neighbors=args.num_neigh)
                save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp3/{args.model}_exp_{dataset}_surv_weights_na_rand_seed_{i}.csv"
                computation_exp.to_csv(save_path, index=False)


def compute_weights(
    explainer: survlime_explainer.SurvLimeExplainer,
    x_test: np.ndarray,
    model: Union[CoxPHSurvivalAnalysis, RandomSurvivalForest],
    num_neighbors: int = 1000
):
    compt_weights = []
    num_pat = num_neighbors
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    for test_point in tqdm(x_test.to_numpy()):
        try:
            b, _ = explainer.explain_instance(
                test_point, predict_chf, verbose=False, num_samples=num_pat
            )

            b = [x[0] for x in b]
        except:
            b = [None] * len(test_point)
        compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=model.feature_names_in_)


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
    args = parser.parse_args()
    main(args)
