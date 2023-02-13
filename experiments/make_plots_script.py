from tqdm import tqdm
import os

import numpy as np
import pandas as pd

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error

def change_name_simulated(name):
    if name=="min":
        return "best"
    elif name=="max":
        return "worst"
    else:
        return name

def change_dataset(name):
    if name=="veterans":
        return "Veteran"
    elif name=="udca":
        return "UDCA"
    elif name=="lung":
        return "LUNG"

def change_name(name):
    if name=="cox":
        return "CoxPH"
    elif name=="rsf":
        return "RSF"
    elif name=="xgb":
        return "XGB"
    elif name=="deepsurv":
        return "DeepSurv"
    elif name=="deephit":
        return "DeepHit"

def create_boxenplot(data):
    median_up = {}
    median_down = {}
    threshold = 0

    for (columnName, columnData) in data.items():
        median_value = np.median(columnData)
        if median_value > threshold:
            median_up[columnName] = median_value
        else:
            median_down[columnName] = median_value

    median_up = dict(
        sorted(median_up.items(), key=lambda item: item[1], reverse=True)
    )
    median_down = dict(
        sorted(median_down.items(), key=lambda item: item[1], reverse=True)
    )

    pal_up = sns.color_palette("Reds_r", n_colors=len(median_up))
    pal_down = sns.color_palette("Blues", n_colors=len(median_down))
    colors_up = {key: val for key, val in zip(median_up.keys(), pal_up)}
    colors_down = {key: val for key, val in zip(median_down.keys(), pal_down)}
    custom_pal = {**colors_up, **colors_down}
    data_reindex = data.reindex(columns=custom_pal.keys())
    data_melt = pd.melt(data_reindex)

    _, ax = plt.subplots(figsize=(11,7))
    ax.tick_params(labelrotation=90)
    p = sns.boxenplot(
        x="variable",
        y="value",
        data=data_melt,
        color="grey",
        #palette=custom_pal,
        ax=ax,
    )
    p.set_xlabel("Features", fontsize= 16)
    p.set_ylabel("SurvLIME value", fontsize= 16)
    p.yaxis.grid(True)
    p.xaxis.grid(True)
    plt.xticks(fontsize=18, rotation=90)
    plt.yticks(fontsize=16, rotation=0)

    return p


def generate_plots_real_datasets():
    results_folder = os.path.join(os.getcwd(), "computed_weights_csv", "exp_real_datasets")

    models = ["cox", "rsf","xgb"]#"deepsurv","deephit"]
    datasets = ["lung", "veterans", "udca"] # "pbc"
    for dataset in datasets:
        for model_type in models:
            file_name = f"{model_type}_exp_{dataset}_surv_weights.csv"
            file_path = os.path.join(results_folder, file_name)
            data = pd.read_csv(file_path)
            data = data.reindex(data.median().sort_values(ascending=False).index, axis=1)
            # if there is a columned named ph.ecog_2.0 rename it to ph.ecog_2
            if "ph.ecog_2.0" in data.columns:
                data.rename(columns={"ph.ecog_2.0": "ph.ecog_2", "ph.ecog_1.0" : "ph.ecog_1"}, inplace=True)
            p = create_boxenplot(data)

            model_name = change_name(model_type)
            dataset_name = change_dataset(dataset)
            p.set_title(f"{model_name} {dataset_name} SurvLIME values", fontsize= 16, fontweight="bold");

            fig_name = f"{model_type}_{dataset}.png"
            fig_path = os.path.join("figures", fig_name)
            plt.savefig(fig_path,  bbox_inches="tight", dpi=200)
            print(f"Figure saved in {fig_path}")


def generate_plots_simulated_experiments():
    for i in [1, 2]:
        for group in ["mean","min","max"]:
            name_file = f"exp_2_cluster_{i}_{group}.csv"
            data_folder = os.path.join(os.getcwd(), "computed_weights_csv", "exp2")
            file_path =  os.path.join(data_folder, name_file)

            data = pd.read_csv(file_path)
            
            p = create_boxenplot(data)
            name = change_name_simulated(group)

            title = f"Cluster {i} {name} values"

            p.set_title(title, fontsize= 18, fontweight="bold");

            fig_name = f"simulated_exp_cluster{i}_{group}_values.png"
            fig_path = os.pardir("figures", fig_name)
            plt.savefig(fig_path,  bbox_inches="tight", dpi=200)
            print(f"Figure saved in {fig_path}")
    

def generate_deepmodels_rds_plots():
    """
    This function generates the plots for the deepsurv_rds experiment
    and saves the result in the figures folder
    """
    results_folder = os.path.join(os.getcwd(), "computed_weights_csv", "exp_deepsurv_rds")
    file_path = os.path.join(results_folder, "exp_deepsurv_rds_surv_weights.csv")
    data = pd.read_csv(file_path)
    
    p = create_boxenplot(data)

    p.set_title(f"DeepSurv RSD SurvLIME values", fontsize= 16, fontweight="bold");

    fig_name = "deepsurv_rds.png"
    fig_path = os.path.join("figures", fig_name)
    plt.savefig(fig_path,  bbox_inches="tight", dpi=200)
    print(f"Figure saved in {fig_path}")

def generate_plot_single_point(cluster):
    data_folder = os.path.join(os.getcwd(), "computed_weights_csv", "exp2")
    if cluster == 1:
        file_name = "center_cluster_1.csv"
        title = "Cluster 1: Values of the coefficients"
        fig_name = "center_cluster_1.png"
    else:
        file_name = "center_cluster_2.csv"
        title = "Cluster 2: Values of the coefficients"
        fig_name = "center_cluster_2.png"
    file_path = os.path.join(data_folder, file_name)
    data = pd.read_csv(file_path, index_col=0)
    p = data.transpose().plot.bar(
        color={"CoxPH": "blue", "SurvLIME": [1.0, 0.55, 0], "Real": "green"},
        rot = 45,
        title = title,
        fontsize = 14
    )
    fig_path = os.path.join("figures", fig_name)
    p.figure.savefig(fig_path)


def plot_experiment_1():
    w_path = "results"
    # Falta escribir el código para la Figure 3 del paper
    coefficients_1 = [10**(-6), 0.1,  -0.15, 10**(-6), 10**(-6)]
    coefficients_2 = [10**(-6), -0.15, 10**(-6), 10**(-6), -0.1]
    mean_dfs = []
    max_dfs  = []
    min_dfs  = []
    for exp, coefficients in zip([1,2], [coefficients_1, coefficients_2]):
        for i in tqdm(range(50)):
            file_name = f"cox_exp_1.{exp}_rand_seed_{i}_surv_weights_na.csv"
            file_path = os.path.join(w_path, file_name)
            compt_weights = pd.read_csv(file_path)
            compt_weights = compt_weights.dropna(axis=0)
            compt_weights.columns = ["One", "Two", "Three", "Four","Five"]
            
            dist = euclidean_distances(compt_weights, np.array(coefficients).reshape(1,-1))
            closest = list(dist).index(min(dist))
            furthest = list(dist).index(max(dist))
            
            mean_dfs.append(compt_weights.mean())
            max_dfs.append(compt_weights.iloc[furthest])
            min_dfs.append(compt_weights.iloc[closest])

                # This will check whether the order of the values are the same. This ensures that the values don"t have high variance
            if i!=0:
                if (compt_weights.iloc[furthest].sort_values(ascending=False).index == max_dfs[-1].sort_values(ascending=False).index).all():
                    pass
                else:
                    print(i)
            
        dfs_together_mean= pd.concat(mean_dfs, axis=1).transpose()
        dfs_together_min= pd.concat(min_dfs, axis=1).transpose()
        dfs_together_max= pd.concat(max_dfs, axis=1).transpose()

        # Order them through the median
        dfs_together_max = dfs_together_max.reindex(dfs_together_max.median().sort_values(ascending=False).index, axis=1)
        dfs_together_min = dfs_together_min.reindex(dfs_together_min.median().sort_values(ascending=False).index, axis=1)
        dfs_together_mean = dfs_together_mean.reindex(dfs_together_mean.median().sort_values(ascending=False).index, axis=1)
        
        # Plot the mean, best and worst performing values of all the repetitions in the form of boxen plots
        fig, ax= plt.subplots(1,3, figsize=(20,5), sharey=True)
        sns.boxenplot(x="variable", y="value", data=pd.melt(dfs_together_mean), ax=ax[0], palette="RdBu")
        ax[0].set_title(f"Cluster {exp} mean values"); ax[0].tick_params(labelrotation=0); ax[0].xaxis.grid(True)
        ax[0].set_ylabel("SurvLIME value")
        ax[0].set_xlabel("Features")

        sns.boxenplot(x="variable", y="value", data=pd.melt(dfs_together_min), ax=ax[1],palette="RdBu")
        ax[1].set_title(f"Cluster {exp} best values"); ax[1].tick_params(labelrotation=0); ax[1].xaxis.grid(True)
        ax[1].set_xlabel("Features")
        ax[1].set_ylabel("SurvLIME value")

        sns.boxenplot(x="variable", y="value", data=pd.melt(dfs_together_max), ax=ax[2],palette="RdBu")
        ax[2].set_title(f"Cluster {exp} worst values"); ax[2].tick_params(labelrotation=0); ax[2].xaxis.grid(True)
        ax[2].set_xlabel("Features")
        ax[2].set_ylabel("SurvLIME value")

        plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        
        # Save the figure
        fig_name = f"exp_1.{exp}_results.png"
        fig_path = os.path.join("figures", fig_name)
        plt.savefig(fig_path, bbox_inches = "tight", dpi=200)
        print(f"Figure saved in {fig_path}")

if __name__ == "__main__":
    # if directory figures does not exist create it
    if not os.path.exists("figures"):
        os.mkdir("figures")
    generate_plots_real_datasets()
