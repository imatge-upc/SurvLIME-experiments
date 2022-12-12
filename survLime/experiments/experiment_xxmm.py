import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set()

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from xgbse import XGBSEKaplanNeighbors
from xgbse._kaplan_neighbors import DEFAULT_PARAMS
from xgbse.metrics import concordance_index

# Our very own survLime!
from survlime.survlime_explainer import SurvLimeExplainer
from survlime import survlime_explainer
from functools import partial

from derma.general.ingestion.data_loader_csv import SurvivalLoader

from script_utils import (nested_stratified_split,
                          data_preprocessing)
import os
import config_os as settings_file

np.random.seed(123456)
get_target = lambda df: df[['event','duration']]
path = '/home/carlos.hernandez/datasets/csvs/data-surv_20220302.csv'

    
def obtain_rsf_kwargs(args) -> dict:
    if args.model == 'rsf':
        if args.event_type=='ss':
            kwargs = {'n_estimators' : 1000,
                        'min_weight_fraction_leaf': 0,
                        'min_samples_split' : 13,
                        'max_features' : 'auto',
                        'max_depth':11}
        elif args.event_type=='dfs':
            kwargs = {'n_estimators' : 1000,
                        'min_weight_fraction_leaf': 0,
                        'min_samples_split' : 11,
                        'max_features' : 'auto',
                        'max_depth':20}
        elif args.event_type=='os':
            kwargs = {'n_estimators' : 1000,
                        'min_weight_fraction_leaf': 0,
                        'min_samples_split' : 2,
                        'max_features' : 'sqrt',
                        'max_depth':19}

    elif args.model == 'xgbse':
        if args.event_type=='ss':
            kwargs = {
                        'gamma': 0.001, 'learning_rate': 0.009970127676395268, 'max_depth': 6,
                        'min_child_weight': 5, 'lambda': 0.0,
                }
        elif args.event_type=='os':
            kwargs = {
                        'gamma': 1e-05, 'learning_rate': 0.009660539112169669, 'max_depth': 16,
                        'min_child_weight': 12, 'lambda': 0.0005,
                }
        elif args.event_type=='dfs':
            kwargs = {
                'gamma': 1e-05, 'learning_rate': 0.009424722066862404, 'max_depth': 8,
                    'min_child_weight': 13, 'lambda': 0.0005,
            }
        # update the default parameters with kwargs
        for k,v in kwargs.items():
            DEFAULT_PARAMS[k] = v
        kwargs = DEFAULT_PARAMS
    else:
        kwargs = {}
    return kwargs



#   keep_cols = clinical_columns +lab_columns
#   settings_file.keep_cols = {'cols' : keep_cols}
    #X_train_previous, X_test, y_train, y_test = train_test_split(X.copy(), np.zeros(len(X)), test_size=0.1, random_state=42)
def main(args):
    X, _, time, event = SurvivalLoader(args.event_type).load_data(path)
    X['duration'] = round(time)

    X['event']    = event
    X.dropna(subset=['duration'],inplace=True)
    X = X[X['duration']>=0]
    splits = nested_stratified_split(X)
    already_trained = False

    kwargs = obtain_rsf_kwargs(args)
    if args.model == 'cox':
        model_pipe = ('coxph', CoxPHSurvivalAnalysis(alpha=0.0001))
    elif args.model == 'xgbse':
        # update the parameters
        model_pipe = ('xgbse', XGBSEKaplanNeighbors(xgb_params = kwargs))
    elif args.model == 'rsf':
        model_pipe = ('rsf', RandomSurvivalForest(
                          **kwargs,
                          random_state=42))
    else:
        AssertionError('Model not implemented')
    for i, (_, train_outer, test_outer) in enumerate(splits):
            y_test = get_target(test_outer); y_train = get_target(train_outer)
            y_train = Surv.from_dataframe(*y_train.columns, y_train)
            y_test = Surv.from_dataframe(*y_test.columns, y_test)

            
            X_train_t, _, X_test_t, columns = data_preprocessing(train_outer,
                                                        train_outer.copy(), test_outer,
                                                        using_dataloaders=False)
            break 

    for repetition in tqdm(range(args.repetitions)):
        try:

            if already_trained:
                pass

            else:
                if args.model == 'xgbse':
                    X_train_model = pd.DataFrame(X_train_t, columns=columns)
                    model_pipe = model_pipe[1].fit(X_train_model, y_train, early_stopping_rounds=12, validation_data=(X_train_model, y_train))
                else:
                    model_pipe = model_pipe[1].fit(X_train_t, y_train)
                    X_train_model = X_train_t
                if args.model == 'xgbse':
                    c_index = concordance_index(y_test, model_pipe.predict(X_test_t))
                    print(c_index)
                already_trained=True

            train_times = [tp[1] for tp in y_train]
            unique_times = list(set(train_times))
            unique_times.sort()

            if args.model == 'xgbse':
                predict_chf= model_pipe.predict
                model_output_times = model_pipe.time_bins
            else:
                predict_chf = partial(model_pipe.predict_cumulative_hazard_function, return_array=True)
                model_output_times = model_pipe.event_times_

            num_neighbors = args.num_neigh
            explainer = SurvLimeExplainer(
                    training_features=X_train_model,
                    traininig_events=[tp[0] for tp in y_train],
                    training_times=[tp[1] for tp in y_train],
                    model_output_times=model_output_times,
                    sample_around_instance=True,
                    random_state=10,
            )

            computation_exp = compute_weights(explainer, X_test_t[:371],
                                              model_pipe, num_neighbors = num_neighbors
                                              , column_names = columns,
                                              predict_chf = predict_chf
                                              )
            save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp4/{args.event_type}/{args.model}_xxmm_surv_{args.event_type}_weights_rand_seed_{repetition}.csv"
            computation_exp.to_csv(save_path, index=False)
        except:
            print('Error in repetition', repetition)
            # print traceback
            import traceback
            print(traceback.print_exc())
            pass

def compute_weights(
    explainer: survlime_explainer.SurvLimeExplainer,
    x_test:  pd.DataFrame,
    model: CoxPHSurvivalAnalysis,
    num_neighbors: int = 1000,
    column_names: list = None,
    predict_chf = None
) -> pd.DataFrame:
    compt_weights = []
    num_pat = num_neighbors
    for test_point in tqdm(x_test):
        try:
            b = explainer.explain_instance(
                test_point, predict_chf, verbose=False, num_samples=num_pat
            )

            compt_weights.append(b)
        except:
            b = [None] * len(test_point)
            compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=column_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obtain SurvLIME results for a given dataset"
    )
    parser.add_argument("--event_type", type=str, default="ss",
                        help="os, dfs or ss"
    )
    parser.add_argument(
        "--model", type=str, default="cox", help="bb model either cox or rsf or xgbse"
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
    args = parser.parse_args()
    main(args)


### Depracated code:
          # pipe = sklearn.pipeline.Pipeline(steps=[
          #     ('TransformToNumeric', TransformToNumeric(**settings_file.transform_to_numeric)), 
          #     ('TransformToDatetime', TransformToDatetime(**settings_file.transform_to_datetime)),
          #     ('TransformToObject', TransformToObject(**settings_file.transform_to_object)),
          #     ('ComputeAge', ComputeAge(**settings_file.compute_age)),
          #     ('tr_tm', LinkTumourPartToParent(**settings_file.link_tumour_part_to_parent)),
          #     ('tr_cb', TransformCbRegression(**settings_file.transform_cb_regression)),
          #     ('tr0', ConvertCategoriesToNaN(**settings_file.convert_categories_to_nan)),
          #     ('tr2', GenderEncoder(**settings_file.gender_encoder)),
          #     ('tr3', AbsentPresentEncoder(**settings_file.absent_present_encoder)),
          #     ("tr4", CategoricalEncoder(**settings_file.categorical_encoder)),
          #     ('tr7', LABEncoder(**settings_file.lab_encoder)),
          #     ('OrdinalEncoder', OrdinalEncoder(**settings_file.ordinal_encoder)),
          #     ('ComputeAJCC', ComputeAJCC(**settings_file.compute_ajcc)),
          #     ('tr5', ExponentialTransformer(**settings_file.exponential_transformer)),
          #     ('KeepColumns', KeepColumns(**settings_file.keep_cols)),
          #     ('RenameLabValues', RenameLabValues(**settings_file.rename_lab_values)),
          #     ('CustomImputer', CustomImputer(strategy='mean')),
          #     ('CustomScaler', CustomScaler()),
          #     model_pipe
          #     ])
          # X_train, X_val, y_train, y_val = train_test_split(X_train_previous.copy(), np.zeros(len(X_train_previous)), test_size=0.2, random_state=repetition)
          # y_train_list = get_target(X_train)
          # 
          # y_train_list = Surv.from_dataframe(*y_train_list.columns, y_train_list)
          # #import ipdb;ipdb.set_trace() 
          # fitted_pipe = pipe.fit(X_train.copy(), y_train_list.copy())
          # X_test_t = fitted_pipe[:-1].transform(X_test)
          # X_train_t = fitted_pipe[:-1].transform(X_train)

          # num_neighbors = args.num_neigh
          # explainer = survlime_explainer.SurvLimeExplainer(X_train_t, y_train_list,
          #                                                  model_output_times=pipe[-1].event_times_)

          # computation_exp = compute_weights(explainer, X_test_t, model_pipe, num_neighbors = num_neighbors)

