import os 
import numpy as np
from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import RandomSurvivalData
import pandas as pd
from pycox.models import CoxPH, DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt
import argparse

def obtain_rds_data():
    # Generate data for the first cluster
    n_points_1 = 1000
    true_coef_1 = [10**(-6), 0.1, -0.15, 10**(-6), 10**(-6)]
    r_1 = 8
    center_1 = [0, 0, 0, 0, 0]
    prob_event_1 = 0.9
    lambda_weibull_1 = 10**(-5)
    v_weibull_1 = 2

    rsd_1 = RandomSurvivalData(
        center=center_1,
        radius=r_1,
        coefficients=true_coef_1,
        prob_event=prob_event_1,
        lambda_weibull=lambda_weibull_1,
        v_weibull=v_weibull_1,
        time_cap=2000,
        random_seed=90,
    )

    X_1, time_to_event_1, delta_1 = rsd_1.random_survival_data(num_points=n_points_1)
    return X_1, time_to_event_1, delta_1



def train_test_split(X_1, delta_1, time_to_event_1):
    n_train_1 = 900
    np.random.seed(90)
    all_idx_1 = np.arange(X_1.shape[0])
    idx_train_1 = np.random.choice(a=all_idx_1, size=n_train_1, replace=False)
    idx_test_1 = [i for i in all_idx_1 if i not in idx_train_1]
    X_train_1 = X_1[idx_train_1, :]
    X_test_1 = X_1[idx_test_1, :]
    time_to_event_train_1 = [time_to_event_1[i] for i in idx_train_1]
    time_to_event_test_1 = [time_to_event_1[i] for i in idx_test_1]
    delta_train_1 = [delta_1[i] for i in idx_train_1]
    delta_test_1 = [delta_1[i] for i in idx_test_1]
    # Transform the data in order to have the DeepHit format
    X_transformed_train = X_train_1.astype('float32')
    X_transformed_test = X_test_1.astype('float32')
    return X_transformed_test, X_transformed_train, delta_test_1, delta_train_1, time_to_event_test_1, time_to_event_train_1



def obtain_model(X_transformed_train, y_train, args, model):
    in_features = X_transformed_train.shape[1]
    num_nodes = [args.num_nodes] * args.num_layers
    num_nodes = [32, 32]
    batch_norm = args.batch_norm
    dropout = args.dropout
    output_bias = args.output_bias
    batch_size = 256
    epochs = 100

    callbacks = [tt.callbacks.EarlyStopping(patience=10)]

    if model =='deepsurv':
        output_nodes = 1
        labtrans = None
    else:
        num_durations = 15
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(*y_train)
        output_nodes = labtrans.out_features

    net_deep_surv = tt.practical.MLPVanilla(in_features, num_nodes, output_nodes, batch_norm,dropout, output_bias=output_bias)
    if model =='deepsurv':
        deep_surv = CoxPH(net_deep_surv, tt.optim.Adam())
    else:
        deep_surv = DeepHitSingle(net_deep_surv, tt.optim.Adam(weight_decay=args.reg), duration_index=labtrans.cuts)
    deep_surv.optimizer.set_lr(args.lr)
    return batch_size, deep_surv, epochs, labtrans



def obtain_c_indexes(X_transformed_test, X_transformed_train, deep_surv, durations_test, events_test, y_deepsurv_train):
    predictions = deep_surv.predict_surv_df(X_transformed_test)
    ev = EvalSurv(predictions, durations_test, events_test, censor_surv='km')
    test_c_index = ev.concordance_td()

    predictions = deep_surv.predict_surv_df(X_transformed_train)
    ev = EvalSurv(predictions, y_deepsurv_train[0], y_deepsurv_train[1] , censor_surv='km')
    train_c_index = ev.concordance_td()
    return test_c_index, train_c_index



def obtain_targets(delta_test_1, delta_train_1, time_to_event_test_1, time_to_event_train_1):
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_df_test = pd.DataFrame(data={'duration': time_to_event_test_1, 'event': delta_test_1})
    durations_test, events_test = get_target(y_df_test)
    y_df_train = pd.DataFrame(data={'duration': time_to_event_train_1, 'event': delta_train_1})
    y_deepsurv_train = get_target(y_df_train)
    return durations_test, events_test, y_deepsurv_train



def deepsurv_rds(args):
    X_1, time_to_event_1, delta_1 = obtain_rds_data()


    X_transformed_test, X_transformed_train, delta_test_1,\
             delta_train_1, time_to_event_test_1, time_to_event_train_1 = train_test_split(X_1, delta_1, time_to_event_1)

    
    for model in ["deephit", "deepsurv"]:
        durations_test, events_test, y_deepsurv_train = obtain_targets(delta_test_1, delta_train_1, time_to_event_test_1, time_to_event_train_1)
        batch_size, deep_surv, epochs, labtrans = obtain_model(X_transformed_train, y_deepsurv_train, args, model)
        
        if model=='deephit':
            y_deepsurv_train = labtrans.fit_transform(*y_deepsurv_train)

        log = deep_surv.fit(
            input=X_transformed_train,
            target=y_deepsurv_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=False
        )
        if model == 'deepsurv':
            deep_surv.compute_baseline_hazards()

        test_c_index, train_c_index = obtain_c_indexes(X_transformed_test, X_transformed_train, deep_surv, durations_test, events_test, y_deepsurv_train)
        print(f"Model: {model}")
        print("Train C-index: ", train_c_index)
        print("Test C-index: ", test_c_index)

    def create_chf(fun):
        def inner(X):
            Y = fun(X)
            return Y.T
        return inner
    predict_chf = create_chf(deep_surv.predict_cumulative_hazards)

    explainer_deepsurv = SurvLimeExplainer(
        training_features=X_transformed_train,
        training_events=delta_train_1,
        training_times=time_to_event_train_1,
        model_output_times=np.sort(np.unique(time_to_event_train_1)),
        random_state=10,
    )

    b_deepsurv = explainer_deepsurv.montecarlo_explanation(
        data=X_transformed_test,
        predict_fn=predict_chf,
        type_fn='cumulative',
        num_samples=1000,
        num_repetitions=args.repetitions,
        verbose=False,
    )

    #import ipdb;ipdb.set_trace()
    save_dir = os.path.join(os.getcwd(), "computed_weights_csv", "exp_deepsurv_rds")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"exp_deepsurv_rds_surv_weights.csv"
    file_directory = os.path.join(save_dir, file_name)
    # transform computation_exp to dataframe
    print('end')
    b_deepsurv_df = pd.DataFrame(b_deepsurv, columns=['one', 'two', 'three', 'four', 'five'])
    b_deepsurv_df.to_csv(file_directory, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--output_bias', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    deepsurv_rds(args)
    print("Done!")
