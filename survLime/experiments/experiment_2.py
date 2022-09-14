import argparse
from typing import List, Union
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from survLime import survlime_explainer
from experiment_1 import create_clusters


def experiment_2(args):

    cluster_0, _ = create_clusters()

    for rep in range(args.repetitions):
        # Experiment 1.1
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
            cluster_0[0], cluster_0[1], test_size=0.1, random_state=rep
        )
        data = []
        labels = []
        for i in range(1, 6):
            data = x_train_1[0 : 100 * i]
            labels = y_train_1[0 : 100 * i]
            _ = experiment(
                [data, labels],
                [x_test_1, y_test_1],
                model_type=args.model,
                exp_name=f"{i}_rand_seed_{rep}"
            )


def experiment(train: List, test: List, model_type: str = "cox", exp_name: str = "1.1"):
    """
    This is going to be the same for all the experiments, we should define it generally
    """
    x_train = train[0]
    y_train = train[1]
    x_test = test[0]
    if model_type == "cox":
        model = CoxPHSurvivalAnalysis(alpha=0.0001)
    elif model_type == "rsf":
        model = RandomSurvivalForest()
    else:
        raise AssertionError(f"The model {model_type} needs to be either [cox or rsf]")

    times_to_fill = list(set([x[1] for x in y_train]))
    times_to_fill.sort()
    columns = ["one", "two", " three", "four", "five"]
    model.fit(x_train, y_train)
    model.feature_names_in_ = columns

    H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)
    explainer = survlime_explainer.SurvLimeExplainer(
        x_train, y_train, model_output_times=model.event_times_
    )
    computation_exp = compute_weights(explainer, x_test, model)
    save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp2/{model_type}_exp_2.{exp_name}_surv_weights_na.csv"
    computation_exp.to_csv(save_path, index=False)
    return computation_exp


def compute_weights(
    explainer: survlime_explainer.SurvLimeExplainer,
    x_test: np.ndarray,
    model: Union[CoxPHSurvivalAnalysis, RandomSurvivalForest],
):
    compt_weights = []
    num_pat = 1000
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    for test_point in tqdm(x_test):
        b, result = explainer.explain_instance(
            test_point, predict_chf, verbose=False, num_samples=num_pat
        )

        b = [x[0] for x in b]
        compt_weights.append(b)
    columns = ["one", "two", "threen", "four", "five"]
    computation_exp = pd.DataFrame(compt_weights, columns=columns)

    return computation_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain SurvLIME results for experiment 1"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="How many times to repeat the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cox",
        help="Which model to use, either cox or rsf"
    )
    args = parser.parse_args()
    experiment_2(args)
