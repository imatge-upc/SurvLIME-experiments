"""
This script is used to execute all the experiments
of the paper "SurvLIMEpy: A Python package implementing SurvLIME"
"""
import argparse
import os

from make_plots_script import (generate_plots_real_datasets,
                             generate_deepmodels_rds_plots,
                            generate_plots_simulated_experiments,
                            generate_plot_single_point)
from experiment_1_montecarlo import experiment_1_cluster_1
from experiment_1_2_montecarlo import experiment_1_cluster_2
from experiment_real_datasets_ml import exp_real_datasets
from deepsurv_rds import deepsurv_rds

if not os.path.exists('figures'):
    os.mkdir('figures')

def execute_experiment(args):
    if args.exp == 'simulated':
        experiment_1_cluster_1(args)
        experiment_1_cluster_2(args)
        generate_plots_simulated_experiments()
        generate_plot_single_point(cluster=1)
        generate_plot_single_point(cluster=2)
    elif args.exp == 'real':
        exp_real_datasets(args)
        generate_plots_real_datasets()
    elif args.exp == 'dl':
        deepsurv_rds(args)
        generate_deepmodels_rds_plots()
    elif args.exp =="all":
        experiment_1_cluster_1(args)
        experiment_1_cluster_2(args)
        exp_real_datasets(args)
        deepsurv_rds(args)
        generate_plots_simulated_experiments()
        generate_plot_single_point(cluster=1)
        generate_plot_single_point(cluster=2)
        generate_plots_real_datasets()
        generate_deepmodels_rds_plots()
    elif args.exp =='only_plot':
        generate_plots_simulated_experiments()
        generate_plot_single_point(cluster=1)
        generate_plot_single_point(cluster=2)
        generate_plots_real_datasets()
        generate_deepmodels_rds_plots()


class exp_params:
    """
    Class to store the hyper-parameters of the experiments for all the models
    """
    def __init__(self) -> None:
        self.repetitions=100

        self.lr = 0.01
        self.reg = 0.1

        self.gamma = 0.1
        self.min_child_weight = 1
        self.aft_loss_distribution = 'normal'
        self.aft_loss_distribution_scale = 1.0

        self.epochs = 100
        self.batch_norm = True
        self.output_bias = True
        self.dropout = 0.5
        self.num_layers = 2

        self.num_nodes = 100
        self.max_features = 100
        self.n_estimators = 100
        self.n_iter = 100
        self.alpha = 0.0005

        self.tol = 0.01
        self.max_depth = 5
        self.min_samples_leaf = 5
        self.min_samples_split = 5
        self.min_weight_fraction_leaf = 0.0

        
if __name__ == "__main__":
    """
    Main function to execute the experiments
    """
    parser = argparse.ArgumentParser(
        description="Choose the experiment"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="either veterans, lung, udca or all",
    )
    parser.add_argument("--exp", default="all", type=str, help="Experiment name")
    parser.add_argument("--model", default="all", help="cox, rsf, xgb, deepsurv, deephit or all")
    
    args = parser.parse_args()
    params = exp_params()
    # add the arguments of the argument parser to the exp_params class
    for arg in vars(args):
        setattr(params, arg, getattr(args, arg))

    execute_experiment(params)



