"""
Experiment for the paper: Put name

We measure how does SurvLIME perform when the number of
neighbours is changed
"""
import warnings
#warnings.filterwarnings("ignore")

import argparse
from typing import List, Union
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from survlime.load_datasets import RandomSurvivalData
from survlime import survlime_explainer

def experiment_neighbours(args):
    
    if args.num_neigh == 'all':
        num_neighbours = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    else:
        num_neighbours = [int(args.num_neigh)]
    model = 'cox'
    for i in range(args.repetitions):
        for num_neigh in tqdm(num_neighbours):
            cluster_1, cluster_2 = create_clusters()

            # Experiment cluster 1 
            x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
                cluster_1[0], cluster_1[1], test_size=0.1, random_state=i
            )
            import ipdb; ipdb.set_trace()
            df = experiment(
                [x_train_1, y_train_1],
                [x_test_1, y_test_1],
                model_type=args.model,
                exp_name=f"1.1_rand_seed_{i}"
            )
             
            # Experiment cluster 2
            x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
                cluster_2[0], cluster_2[1], test_size=0.1, random_state=i
            )
            df = experiment(
                [x_train_2, y_train_2],
                [x_test_2, y_test_2],
                model_type=args.model,
                exp_name=f"1.2_rand_seed_{i}"
            )


def experiment(train: List, test: List, model_type: str = "cox", exp_name: str = "3.1", num_neighbours: int = 1000):
    """
    This is going to be the same for all the experiments, we should define it generally

    """
    x_train = train[0]
    y_train = train[1]
    x_test = test[0]

    model = CoxPHSurvivalAnalysis(alpha=0.0001)

    times_to_fill = list(set([x[1] for x in y_train]))
    times_to_fill.sort()

    events_train = [x[0] for x in y_train]
    times_train  = [x[1] for x in y_train]

    columns = ["one", "two", " three", "four", "five"]
    model.fit(x_train, y_train)
    model.feature_names_in_ = columns

    # H0 = model.cum_baseline_hazard_.y.reshape(len(times_to_fill), 1)
    explainer = survlime_explainer.SurvLimeExplainer(
        x_train, events_train, times_train, model_output_times=times_to_fill)
    
    for neighbours in range(1, args.num_neigh+2, 20)
    computation_exp = compute_weights(explainer, x_test, model, num_pat = neighbours)
    save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp_neigh/{model_type}_exp_surv_weights_{num_neighbours}.csv"
    return computation_exp


def compute_weights(
    explainer: survlime_explainer.SurvLimeExplainer,
    x_test: np.ndarray,
    model: Union[CoxPHSurvivalAnalysis, RandomSurvivalForest],
    num_neighbours: int=1000,
):
    compt_weights = []
    num_pat = num_neighbours
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    try:
        b = explainer.explain_instance(
            test_point, predict_chf, verbose=False, num_samples=num_pat
        )

    except:
        b = [None] * len(test_point)
    compt_weights.append(b)
    columns = ["one", "two", "three", "four", "five"]
    computation_exp = pd.DataFrame(compt_weights, columns=columns)

    return computation_exp


def create_clusters():
    """
    Creates the clusters proposed in the paper: https://arxiv.org/pdf/2003.08371.pdf

    Returns:
    cluster 0: list[Data, target]
    cluster 1: List[Data, target]
    """

    # These values are shared among both clusters
    radius = 8
    num_points = 1000
    prob_event = 0.9
    lambda_weibull = 10 ** (-6)
    v_weibull = 2

    # First cluster
    center = [0, 0, 0, 0, 0]
    coefficients = [10 ** (-6), 0.1, -0.15, 10 ** (-6), 10 ** (-6)]
    rds = RandomSurvivalData(
        center,
        radius,
        coefficients,
        prob_event,
        lambda_weibull,
        v_weibull,
        random_seed=23,
    )
    X_0, T_0, delta_0 = rds.random_survival_data(num_points)
    z_0 = [(d, int(t)) for d, t in zip(delta_0, T_0)]
    y_0 = np.array(z_0, dtype=[("delta", np.bool_), ("time_to_event", np.float32)])

    # From page 6 of the paper (I think)
    center = [4, -8, 2, 4, 2]
    coefficients = [10 ** (-6), -0.15, 10 ** (-6), 10 ** (-6), -0.1]
    rds = RandomSurvivalData(
        center,
        radius,
        coefficients,
        prob_event,
        lambda_weibull,
        v_weibull,
        random_seed=23,
    )
    X_1, T_1, delta_1 = rds.random_survival_data(num_points)
    z_1 = [(d, int(t)) for d, t in zip(delta_1, T_1)]
    y_1 = np.array(z_1, dtype=[("delta", np.bool_), ("time_to_event", np.float32)])

    return [X_0, y_0], [X_1, y_1]


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
        help="Which model to use, either cox or rsf or both",
    )
    parser.add_argument(
        "--num_neigh",
        default=1000,
        help="Number of neighbours to use for the explanation",
    )
    args = parser.parse_args()
    experiment_1(args)
