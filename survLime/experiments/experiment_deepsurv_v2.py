"""Pipeline training script."""
import os
os.chdir("..")
from tqdm import tqdm

# DL for survival
import pycox
from pycox.models import CoxPH, DeepHitSingle

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

import config_os as settings_file
from survlime.survlime_explainer import SurvLimeExplainer


import torch
np.random.seed(1234)
_ = torch.manual_seed(1234)

keep_cols = ['patient_gender', 'patient_eye_color', 'patient_phototype', 'cutaneous_biopsy_breslow', 'cutaneous_biopsy_mitotic_index', 'cutaneous_biopsy_associated_nevus', 'cutaneous_biopsy_vascular_invasion', 'cutaneous_biopsy_regression', 'cutaneous_biopsy_lymphatic_invasion', 'cutaneous_biopsy_ulceration', 'cutaneous_biopsy_neurotropism', 'cutaneous_biopsy_satellitosis', 'mc1r', 'LAB1300', 'LAB1301', 'LAB1307', 'LAB1309', 'LAB1311', 'LAB1313', 'LAB1314', 'LAB1316', 'LAB2404', 'LAB2405', 'LAB2406', 'LAB2407', 'LAB2419', 'LAB2422', 'LAB2467', 'LAB2469', 'LAB2476', 'LAB2498', 'LAB2544', 'LAB2679', 'LAB4176', 'high_abv_frequency', 'height', 'weight', 'nevi_count', 'nca', 'nca_count', 'blue_nevi_count', 'time_smoking', 'cigars_per_day', 'amount_sun_exposure', 'braf_mutation', 'cdkn2a_mutation', 'decision_dx',
    'age', 'primary_tumour_location_coded_acral', 'primary_tumour_location_coded_head and neck', 'primary_tumour_location_coded_lower limbs', 'primary_tumour_location_coded_upper limbs', 'primary_tumour_location_coded_mucosa', 'cutaneous_biopsy_predominant_cell_type_fusocellular', 'cutaneous_biopsy_predominant_cell_type_pleomorphic', 'cutaneous_biopsy_predominant_cell_type_sarcomatoid', 'cutaneous_biopsy_predominant_cell_type_small_cell', 'cutaneous_biopsy_predominant_cell_type_spindle', 'cutaneous_biopsy_histological_subtype_acral_lentiginous', 'cutaneous_biopsy_histological_subtype_desmoplastic', 'cutaneous_biopsy_histological_subtype_lentiginous_malignant', 'cutaneous_biopsy_histological_subtype_mucosal', 'cutaneous_biopsy_histological_subtype_nevoid', 'cutaneous_biopsy_histological_subtype_nodular', 'cutaneous_biopsy_histological_subtype_spitzoid', 'patient_hair_color_black', 'patient_hair_color_blond', 'patient_hair_color_red']

keep_cols = ['patient_gender', 'cutaneous_biopsy_breslow']

def train_model(args, data, label):
    """ Documentation train loop
    Arguments:
    args: Arguments from the parser
    data: Data to train
    label: Label to train
    """
   #y_train = get_target(label[0])
   #y_val   = get_target(label[1])

    # Transform data into numpy arrays
  # leave = [(col, None) for col in data[0].columns]
  # x_mapper = DataFrameMapper(leave)
  # x_train = x_mapper.fit_transform(data[0]).astype('float32')
  # x_val = x_mapper.transform(data[1]).astype('float32')

    # Model Hyperparms
    num_nodes = [args.num_nodes for i in range(args.num_layers)]

    batch_norm = args.batch_norm; dropout = args.dropout ;output_bias = args.output_bias
    batch_size = 256; epochs = 2000
    callbacks = [tt.callbacks.EarlyStopping()]

    in_features = data[0].shape[1]
    if args.model == 'deepsurv':
        out_features = 1
        y_train = label[0]
    elif args.model =='deephit':
        num_durations = 20
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(label[0][0], label[0][1])
        out_features = labtrans.out_features
    
    # Instantiate the model
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    
    if args.model == 'deepsurv':
        model = pycox.models.CoxPH(net, tt.optim.Adam(weight_decay=args.reg))
    elif args.model == 'deephit':
        model = DeepHitSingle(net, tt.optim.Adam(weight_decay=args.reg), duration_index=labtrans.cuts)
    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    model.optimizer.set_lr(args.lr)

    x_train = data[0].astype('float32')
    x_val = data[1].astype('float32')

    #val = x_val, label[1]

    # Train!
    log = model.fit(input=x_train, target=y_train, batch_size=batch_size,
                    epochs=epochs, callbacks=callbacks, verbose=False,
                    val_data=(x_val, label[1])
                        )
    # if the output of the model is of length one
    if out_features == 1:
        model.compute_baseline_hazards()

    # Compute the C-index
    surv = model.predict_surv_df(x_train)
    ev = EvalSurv(surv, y_train[0], y_train[1], censor_surv='km')
    c_index = ev.concordance_td('antolini')
    print(f"C-index: {c_index:.3f}")

    return model

def drop_columns(X_test_t, X_train_t, X_val_t, columns):
    drop_cols = [i for i in columns if i not in keep_cols]
    drop_cols = [columns.to_list().index(col) for col in drop_cols]
    X_train_t = np.delete(X_train_t, drop_cols, axis=1)
    X_val_t = np.delete(X_val_t, drop_cols, axis=1)
    X_test_t = np.delete(X_test_t, drop_cols, axis=1)
    return X_test_t, X_train_t, X_val_t



def main(args):
    path = '/home/carlos.hernandez/datasets/csvs/data-surv_20220302.csv'

    X, _, time, event = SurvivalLoader(args.event_type).load_data(path)
    X['duration'] = round(time)
    X['event']    = event
    X.dropna(subset=['duration'],inplace=True)
    X = X[X['duration']>=0]

    label = X[['duration', 'event']]

    #_train_first, X_test, y_train_first, y_test = train_test_split(X.copy(), label.copy(), test_size=0.1, random_state=42)
    
    get_target = lambda df: (df['duration'].values, df['event'].values)
    splits = nested_stratified_split(X)
#   X_train_previous, X_test, y_train, y_test = train_test_split(X.copy(), np.zeros(len(X)), test_size=0.1, random_state=42)
    for repetition in tqdm(range(args.repetitions)):
        for i, (inner_splits, train_outer, test_outer) in enumerate(splits):
                y_test = get_target(test_outer); y_train = get_target(train_outer)
                #y_train = Surv.from_dataframe(*['duration','event'], y_train)
                X_train_t, X_val_t, X_test_t, columns = data_preprocessing(train_outer,
                                                            train_outer.copy(), test_outer,
                                                         using_dataloaders=False)
                break

        # Delete the column of X_train_t and X_val_t and X_test_t that are not in keep_cols
    
        #X_train_t, X_val_t, X_test_t = drop_columns(X_test_t, X_train_t, X_val_t, columns)

        
        # Save X_traint_t, X_val_t, X_test_t, y_train, y_test in a single .csv file
      # X_train_t.to_csv('X_train_t.csv', index=False)
      # X_val_t.to_csv('X_val_t.csv', index=False)
      # X_test_t.to_csv('X_test_t.csv', index=False)
      # y_csv = pd.DataFrrme(y_train, columns=['duration', 'event'])

      # y_csv.to_csv('y_train.csv', index=False)
        # print the shape of x_train_t and x_val_t and x_test_t
        print(X_train_t.shape, X_val_t.shape, X_test_t.shape)
        splits = []; labels = []
        splits.append([X_train_t, X_train_t, X_test_t])
        labels.append([y_train, y_train, y_test])
        in_features = X_train_t.shape[1]

        data = splits[0]; label = labels[0]

        # Obtain labels
        model =  train_model(args, data, label) # We replace validation with train
          
        # evaluate model with test data
        surv = model.predict_surv_df(data[2])
        ev = EvalSurv(surv, label[2][0], label[2][1], censor_surv='km')
        c_index = ev.concordance_td('antolini')
        print(f"C-index Test set: {c_index:.3f}")


        train_times = [x for x in y_train[0]]
        y_train_fixed = (y_train[0], [True if x==1 else False for x in y_train[1]])
        unique_times = list(set(train_times))
        unique_times.sort()
        
        if args.model == 'deepsurv':
            predict_chf = model.predict_surv
           #chosen_H0 = model.baseline_hazards_.to_numpy()
           #chosen_H0 = np.reshape(chosen_H0, (chosen_H0.shape[0],1))
            #print(chosen_H0.shape)
            chosen_H0 = None
            model_output_times = unique_times
        else:
            predict_chf = model.predict_surv
            model_output_times = model.duration_index
            chosen_H0 = None
        # concatenate X_train_t and X_test_t
        print(X_train_t.shape, X_test_t.shape)
       #X_train_t = np.concatenate((X_train_t, X_test_t), axis=0)
        print(X_train_t.shape)
        explainer = SurvLimeExplainer(
                training_features=X_train_t,
                training_events=y_train_fixed[1],
                training_times=y_train_fixed[0],
                H0= chosen_H0,
                model_output_times=model_output_times,
                sample_around_instance=True,
                random_state=10,
        )
        computation_exp = compute_weights(explainer, data[2][:371], model,
                                          predict_chf = predict_chf,
                                          column_names = columns,
                                          num_neighbors = args.num_neighbours
                                          )
        computation_exp.columns = X_train_t.columns
        
        save_path = f"/home/carlos.hernandez/PhD/survlime-paper/survLime/computed_weights_csv/exp4/DeepSurv/{args.event_type}/DeepSurv_xxmm_surv_{args.event_type}_weights_rand_seed_{repetition}.csv"
        #computation_exp.to_csv(save_path, index=False)

    # Print a line of # and then "experiment ended"
    print("#" * 100)
    print("Experiment ended")
    print("#" * 100)

def compute_weights(
    explainer: SurvLimeExplainer,
    x_test: np.ndarray,
    model: None,
    num_neighbors: int = 1000,
    column_names: list = None,
    predict_chf = None
) -> pd.DataFrame:
    """
    Compute the weights for the given data points
    :param explainer: the explainer object
    :param x_test: the data points for which the weights should be computed
    :param model: the model
    :param num_neighbors: the number of neighbors to us
    :param column_names: the column names

    :return: the weights
    """
    compt_weights = []
    num_pat = num_neighbors
    for test_point in tqdm(x_test):
        try:
            b = explainer.explain_instance(
                test_point, predict_chf, verbose=False, num_samples=num_pat,
                type_fn='survival'
            )

            compt_weights.append(b)
        except:
            import traceback
            print(traceback.print_exc())
            import ipdb;ipdb.set_trace()
            b = [None] * len(test_point)
            compt_weights.append(b)

    return pd.DataFrame(compt_weights, columns=column_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add model arg
    parser.add_argument('--model', type=str, default='deepsurv')
    parser.add_argument('--event_type', type=str, default='os', help='Event type to predict')
    parser.add_argument('--num_neighbours', type=int, default=1000, help='Number of neighbours to use')
    # add repetitions
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_nodes', type=int, default=64)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--output_bias', type=bool, default=False)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
