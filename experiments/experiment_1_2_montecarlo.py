from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import RandomSurvivalData
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.metrics import pairwise_distances
import os
import pandas as pd


def obtain_fitted_model(X_1, delta_1, time_to_event_1):
    # Train test split for the first cluster
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
    z_train_1 = [(d, t) for d, t in zip(delta_train_1, time_to_event_train_1)]
    y_train_1 = np.array(z_train_1, dtype=[("delta", np.bool_), ("time_to_event", np.float32)])
    # Fit a Cox model
    cox = CoxPHSurvivalAnalysis()
    cox.fit(X_train_1, y_train_1)
    return X_test_1, X_train_1, cox, delta_train_1, time_to_event_train_1



def save_mid_experiment(coeff_max_distance, coeff_mean_distance, coeff_min_distance, col_names, file_directory_max, file_directory_mean, file_directory_min, n_features_1):
    coeff_min_distance_np = np.array(coeff_min_distance).reshape(-1, n_features_1)
    coeff_mean_distance_np = np.array(coeff_mean_distance).reshape(-1, n_features_1)
    coeff_max_distance_np = np.array(coeff_max_distance).reshape(-1, n_features_1)

    df_to_save_min = pd.DataFrame(coeff_min_distance_np, columns=col_names)
    df_to_save_mean = pd.DataFrame(coeff_mean_distance_np, columns=col_names)
    df_to_save_max = pd.DataFrame(coeff_max_distance_np, columns=col_names)

    if not os.path.exists(file_directory_min):
        df_load_min = pd.DataFrame(columns=col_names)
    else:
        df_load_min = pd.read_csv(file_directory_min)

    if not os.path.exists(file_directory_mean):
        df_load_mean = pd.DataFrame(columns=col_names)
    else:
        df_load_mean = pd.read_csv(file_directory_mean)

    if not os.path.exists(file_directory_max):
        df_load_max = pd.DataFrame(columns=col_names)
    else:
        df_load_max = pd.read_csv(file_directory_max)

    df_load_min = pd.read_csv(file_directory_min)
    df_load_mean = pd.read_csv(file_directory_mean)
    df_load_max = pd.read_csv(file_directory_max)

    df_min = pd.concat([df_load_min, df_to_save_min])
    df_mean = pd.concat([df_load_mean, df_to_save_mean])
    df_max = pd.concat([df_load_max, df_to_save_max])

    df_min.to_csv(file_directory_min, columns=col_names, index=False)
    df_mean.to_csv(file_directory_mean, columns=col_names, index=False)
    df_max.to_csv(file_directory_max, columns=col_names, index=False)



def obtain_random_data():
    n_points_1 = 1000
    true_coef_1 = [10**(-6), -0.15, 10**(-6), 10**(-6), -0.1]
    r_1 = 8
    center_1 = [4, -8, 2, 4, 2]
    prob_event_1 = 0.9
    lambda_weibull_1 = 10**(-5)
    v_weibull_1 = 2
    n_features_1 = len(true_coef_1)

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
    return X_1, n_features_1, true_coef_1, time_to_event_1, delta_1, center_1, true_coef_1

data_folder = os.path.join(os.getcwd(), "computed_weights_csv", "exp2")
file_name_min = "exp_2_cluster_2_min.csv"
file_name_mean = "exp_2_cluster_2_mean.csv"
file_name_max = "exp_2_cluster_2_max.csv"
file_name_point = "center_cluster_2.csv"
file_directory_min = os.path.join(data_folder, file_name_min)
file_directory_mean = os.path.join(data_folder, file_name_mean)
file_directory_max = os.path.join(data_folder, file_name_max)
file_directory_point = os.path.join(data_folder, file_name_point)

def explain_point(point, train_features, train_events, train_times, unique_times, pred_fn, num_samples, real_coefficients, model_coefficients, feature_names):
    explainer = SurvLimeExplainer(
        training_features=train_features,
        training_events=train_events,
        training_times=train_times,
        model_output_times=unique_times,
        random_state=20,
    )

    b = explainer.explain_instance(
            data_row=point,
            predict_fn=pred_fn,
            num_samples=num_samples,
            verbose=False,
        )
    data = np.empty((3, len(feature_names)))
    data[0] = real_coefficients
    data[1] = model_coefficients
    data[2] = b
    data_pd = pd.DataFrame(data, index=["Real", "CoxPH", "SurvLIME"], columns=feature_names)
    data_pd.to_csv(file_directory_point)

def experiment_1_cluster_2(args):
    """
    Second experiment for the simulated data
    These experiments correspond to the section 4.1 of the paper
    """
    X_1, n_features_1, true_coef_1, time_to_event_1, delta_1, center_1, cluster_coefficients = obtain_random_data()

    X_test_1, X_train_1, cox, delta_train_1, time_to_event_train_1 = obtain_fitted_model(X_1, delta_1, time_to_event_1)

    # SurvLime for COX
    col_names = ["one", "two", "three", "four", "five"]
    num_repetitions = args.repetitions
    num_samples = 1000
    total_test_1 = X_test_1.shape[0]
    true_coef_np = np.array(true_coef_1).reshape(1, -1)

    coeff_min_distance = []
    coeff_mean_distance = []
    coeff_max_distance = []
    i_individual = 0
    explain_point(center_1, X_train_1, delta_train_1, time_to_event_train_1, cox.event_times_, cox.predict_cumulative_hazard_function, num_samples, cluster_coefficients, cox.coef_, col_names)
    while i_individual < total_test_1:
        print(f"Workin on individual {i_individual} out of {total_test_1}")
        B_individual = np.full(shape=(num_repetitions, n_features_1), fill_value=np.nan)
        individual = X_test_1[i_individual] 
        for i_sim in range(num_repetitions):
            explainer_cox = SurvLimeExplainer(
                training_features=X_train_1,
                training_events=delta_train_1,
                training_times=time_to_event_train_1,
                model_output_times=cox.event_times_,
                random_state=i_sim,
            )
            b_cox = explainer_cox.explain_instance(
                data_row=individual,
                predict_fn=cox.predict_cumulative_hazard_function,
                num_samples=num_samples,
                verbose=False,
            )
            B_individual[i_sim] = b_cox
        dist_matrix = pairwise_distances(true_coef_np, B_individual)
        idx_min = np.argmin(dist_matrix)
        idx_max = np.argmax(dist_matrix)
        b_min = B_individual[idx_min]
        b_mean = np.mean(B_individual, axis=0)
        b_max = B_individual[idx_max]
        coeff_min_distance.append(b_min)
        coeff_mean_distance.append(b_mean)
        coeff_max_distance.append(b_max)

        if i_individual % 5 == 0 or i_individual == (total_test_1 - 1):
            save_mid_experiment(coeff_max_distance, coeff_mean_distance, coeff_min_distance, col_names, file_directory_max, file_directory_mean, file_directory_min, n_features_1)
            coeff_min_distance = []
            coeff_mean_distance = []
            coeff_max_distance = []
            print(f"\tSaved individual {i_individual}")
        i_individual += 1

