"""Pipeline training script."""
import os
os.chdir("..")
from tqdm import tqdm

# DL for survival
import pycox

from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.evaluation import EvalSurv

import torchtuples as tt
import argparse
import numpy as np
import pandas as pd


from sklearn.model_selection import KFold, train_test_split

from sksurv.util import Surv
import wandb
import sklearn

from derma.general.ingestion.data_loader_csv import SurvivalLoader


from script_utils import (nested_stratified_split,
                          data_preprocessing)
from derma.general.preprocessing.transformers import (TransformToNumeric, 
                                                      TransformToDatetime, 
                                                      ComputeAge,
                                                      TransformToObject,
                                                      KeepColumns,
                                                      ComputeAJCC,
                                                      LinkTumourPartToParent,
                                                      TransformCbRegression,
                                                      ConvertCategoriesToNaN,
                                                      ExponentialTransformer,
                                                      RenameLabValues,
                                                      CustomScaler,
                                                      CustomImputer)

from derma.general.preprocessing.encoders import (OrdinalEncoder,
                                                  GenderEncoder,
                                                  AbsentPresentEncoder,
                                                  LABEncoder,
                                                  CategoricalEncoder)

import config_os as settings_file
from survLime import survlime_explainer

def train_model(args, batch_norm, batch_size, callbacks, data,
                dropout, epochs, get_target, in_features, label,
                num_nodes, out_features, output_bias, y_test, y_train, y_val):
    """ Documentation train loop
    Arguments:
    args: Arguments from the parser
    batch_norm: Boolean to use batch normalization
    batch_size: Batch size
    callbacks: Callbacks to use
    data: Data to train
    dropout: Dropout to use
    """
   #y_train = get_target(label[0])
   #y_val   = get_target(label[1])

    # Transform data into numpy arrays
  # leave = [(col, None) for col in data[0].columns]
  # x_mapper = DataFrameMapper(leave)
  # x_train = x_mapper.fit_transform(data[0]).astype('float32')
  # x_val = x_mapper.transform(data[1]).astype('float32')
    x_train = data[0].astype('float32')
    x_val = data[1].astype('float32')
    val = x_val, y_val

    # Instantiate the model
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

    model = pycox.models.CoxPH(net, tt.optim.Adam(weight_decay=args.reg))
    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    model.optimizer.set_lr(args.lr)

    # Train!
    log = model.fit(input=x_train, target=y_train, batch_size=batch_size,
                    epochs=epochs, callbacks=callbacks, verbose=False,
                        val_data=val, val_batch_size=batch_size)

    model.compute_baseline_hazards()
    # Compute the C-index
    surv = model.predict_surv_df(x_train)
    ev = EvalSurv(surv, y_train[0], y_train[1], censor_surv='km')
    c_index = ev.concordance_td('antolini')
    print(f"C-index: {c_index:.3f}")

    return model

def obtain_pipe():
    pipe = sklearn.pipeline.Pipeline(steps=[
        ('TransformToNumeric', TransformToNumeric(**settings_file.transform_to_numeric)), 
        ('TransformToDatetime', TransformToDatetime(**settings_file.transform_to_datetime)),
        ('TransformToObject', TransformToObject(**settings_file.transform_to_object)),
        ('ComputeAge', ComputeAge(**settings_file.compute_age)),
        ('tr_tm', LinkTumourPartToParent(**settings_file.link_tumour_part_to_parent)),
        ('tr_cb', TransformCbRegression(**settings_file.transform_cb_regression)),
        ('tr0', ConvertCategoriesToNaN(**settings_file.convert_categories_to_nan)),
        ('tr2', GenderEncoder(**settings_file.gender_encoder)),
        ('tr3', AbsentPresentEncoder(**settings_file.absent_present_encoder)),
        ("tr4", CategoricalEncoder(**settings_file.categorical_encoder)),
        ('tr7', LABEncoder(**settings_file.lab_encoder)),
        ('OrdinalEncoder', OrdinalEncoder(**settings_file.ordinal_encoder)),
        ('ComputeAJCC', ComputeAJCC(**settings_file.compute_ajcc)),
        ('tr5', ExponentialTransformer(**settings_file.exponential_transformer)),
        ('KeepColumns', KeepColumns(**settings_file.keep_cols)),
        ('RenameLabValues', RenameLabValues(**settings_file.rename_lab_values)),
        ('CustomImputer', CustomImputer(strategy='mean')),
        ('CustomScaler', CustomScaler()),
        ])
    return pipe

def main(args):
    path = '/home/carlos.hernandez/datasets/csvs/data-surv_20220302.csv'

    X, _, time, event = SurvivalLoader(args.event_type).load_data(path)
    X['duration'] = round(time)
    X['event']    = event
    X.dropna(subset=['duration'],inplace=True)
    X = X[X['duration']>=0]
    pipe = obtain_pipe()

    label = X[['duration', 'event']]

    #_train_first, X_test, y_train_first, y_test = train_test_split(X.copy(), label.copy(), test_size=0.1, random_state=42)
    
    get_target = lambda df: (df['duration'].values, df['event'].values)
    splits = nested_stratified_split(X)
    X_train_previous, X_test, y_train, y_test = train_test_split(X.copy(), np.zeros(len(X)), test_size=0.1, random_state=42)
    for repetition in tqdm(range(args.repetitions)):

        for i, (inner_splits, train_outer, test_outer) in enumerate(splits):
                y_test = get_target(test_outer); y_train = get_target(train_outer)
                #y_train = Surv.from_dataframe(*['duration','event'], y_train)
                X_train_t, X_val_t, X_test_t, columns = data_preprocessing(train_outer,
                                                            train_outer.copy(), test_outer,
                                                            using_dataloaders=False)
                break 
        # Delete the column of X_train_t and X_val_t and X_test_t that have the same index
        # as 'nodal_met', 'visceral_met' and 'cutaneous_met' in columns
        drop_cols = ['nodal_met', 'visceral_met', 'cutaneous_met'];
        import ipdb;ipdb.set_trace()
        drop_cols = [columns.to_list().index(col) for col in drop_cols]
        X_train_t = np.delete(X_train_t, drop_cols, axis=1)
        X_val_t = np.delete(X_val_t, drop_cols, axis=1)
        X_test_t = np.delete(X_test_t, drop_cols, axis=1)
        splits = []; labels = []
       #X_train, X_val, y_train, y_val = train_test_split(X_train_first.copy(), y_train_first, test_size=0.2, random_state=repetition)
       #X_train_t = pipe.fit_transform(X_train.copy(), y_train)
       #
       #X_val_t   = pipe.transform(X_val.copy())
       #X_test_t  = pipe.transform(X_test.copy())   

        splits.append([X_train_t, X_val_t, X_test_t])
        labels.append([y_train, y_train, y_test])
        in_features = X_train_t.shape[1]

        if args.num_layers==1:
            num_nodes = [args.num_nodes]
        elif args.num_layers==2:
            num_nodes = [args.num_nodes, args.num_nodes]
        elif args.num_layers==3:
            num_nodes = [args.num_nodes, args.num_nodes, args.num_nodes]
        elif args.num_layers==4:
            num_nodes = [args.num_nodes, args.num_nodes, args.num_nodes, args.num_nodes]

        out_features = 1
        batch_norm = args.batch_norm; dropout = args.dropout ;output_bias = args.output_bias
        batch_size = 256; epochs = 20
        callbacks = [tt.callbacks.EarlyStopping()]
        
        data = splits[0]
        label = labels[0]

        # Obtain labels
        model =  train_model(args, batch_norm, batch_size, callbacks, data, dropout, epochs, get_target, in_features, label,
                             num_nodes, out_features, output_bias, y_test, y_train, y_train) # We replace validation with train

        train_times = [x for x in y_train[0]]
        y_train_fixed = (y_train[0], [True if x==1 else False for x in y_train[1]])
        unique_times = list(set(train_times))
        unique_times.sort()

        import ipdb; ipdb.set_trace()
        explainer = survlime_explainer.SurvLimeExplainer(X_train_t, y_train_fixed,
                                                         model_output_times=unique_times)
        
        computation_exp = compute_weights(explainer, data[2], model, num_neighbors = 1000)
        computation_exp.columns = X_train_t.columns

        
        save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp4/DeepSurv/{args.event_type}/DeepSurv_xxmm_surv_{args.event_type}_weights_rand_seed_{repetition}.csv"
        computation_exp.to_csv(save_path, index=False)

    # Print a line of # and then "experiment ended"
    print("#" * 100)
    print("Experiment ended")
    print("#" * 100)

def compute_weights(
    explainer: survlime_explainer.SurvLimeExplainer,
    x_test: np.ndarray,
    model: None,
    num_neighbors: int = 1000,
):
    """
    Computes the weights for the given test data.
    :param explainer: The explainer to use.
    :param x_test: The test data.
    :param model: The model to use.

    """
    compt_weights = []
    num_pat = num_neighbors
    predict_chf = model.predict_cumulative_hazards
    for test_point in tqdm(x_test):
        try:
            b, _ = explainer.explain_instance(
                test_point, predict_chf, verbose=True, num_samples=num_pat
            )

            b = [x[0] for x in b]
        except:
            print('bad yuyu')
            b = [None] * len(test_point)
        compt_weights.append(b)

    computation_exp = pd.DataFrame(compt_weights)
    return computation_exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_type', type=str, default='dfs', help='Event type to predict')
    # add repetitions
    parser.add_argument('--repetitions', type=int, default=100, help='Number of repetitions')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_nodes', type=int, default=128)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output_bias', type=bool, default=True)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
