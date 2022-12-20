from typing import Union, List, Callable
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
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from xgbse.metrics import concordance_index
from xgbse._kaplan_neighbors import DEFAULT_PARAMS

from survlime.survlime_explainer import SurvLimeExplainer
from survlime.load_datasets import Loader


def main(args):
    if args.dataset == "all":
        datasets = ["veterans", "udca", "lung", "pbc"]
    else:
        datasets = [args.dataset]
    if args.model != "all":
        models = [args.model]
    else:
        models = ["cox", "rsf",'xgb']
    for model in models:
        args.model = model
        for dataset in datasets:
            for i in tqdm(range(args.repetitions)):
                loader = Loader(dataset_name=dataset)
                x, events, times = loader.load_data()

                train, _, test = loader.preprocess_datasets(x, events, times, random_seed=i)

                if args.model == "cox":
                    model = CoxPHSurvivalAnalysis(alpha=0.0001)
                    model_output_times = model.event_times_
                    type_fn = 'cumulative'
                elif args.model == "rsf":
                    model = RandomSurvivalForest()
                    model_output_times = model.event_times_
                    type_fn = 'cumulative'
                elif args.model == 'xgb':
                    DEFAULT_PARAMS =  {'objective': 'survival:aft',
                                                  'eval_metric': 'aft-nloglik',
                                                  'aft_loss_distribution': 'normal',
                                                  'aft_loss_distribution_scale': 1.20,
                                                  'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 10}
                    model = XGBSEKaplanNeighbors(DEFAULT_PARAMS, n_neighbors=50)
                    predict_chf = model.predict
                    type_fn = 'survival'
                else:
                    raise AssertionError
                model.fit(train[0], train[1])
                # plot concordance index on test set
                preds = model.predict(train[0])
                c_index = concordance_index(train[1], preds)
                print(f"Concordance index on test set: {c_index:.3f}")
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
                        sample_around_instance=True,
                        random_state=10,
                )

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
        try:
            b = explainer.explain_instance(
                test_point, predict_chf,
                verbose=False, num_samples=num_pat,
                type_fn = type_fn
            )

        except:
            import traceback
            traceback.print_exc()
            print(f"Error in the computation of the weights in patient {i}")
            b = [None] * len(test_point)
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
    args = parser.parse_args()
    main(args)
