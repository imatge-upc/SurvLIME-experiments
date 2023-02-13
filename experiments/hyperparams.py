### Random survival forest
rsf_lung = {'max_depth': 3, 'min_samples_leaf': 17, 'min_samples_split':9,
            'n_estimators': 1000, 'min_weight_fraction_leaf':0}

rsf_veterans = {'max_depth': 5, 'min_samples_leaf': 8, 'min_samples_split': 15,
            'n_estimators': 100, 'min_weight_fraction_leaf':0.05}

rsf_udca = {'max_depth': 4, 'min_samples_leaf': 10, 'min_samples_split': 5,
            'n_estimators': 800, 'min_weight_fraction_leaf':0}

### XGBoost Survival Embeddings
xgb_lung = {'alpha': 1e-05, 'lr': 0.05,
             'max_depth': 15, 'min_child_weight': 1}

xgb_veterans = {'alpha': 0.0001, 'lr': 0.1,
             'max_depth': 14, 'min_child_weight': 1}

xgb_udca = {'alpha': 0.0005, 'lr': 0.01,
             'max_depth': 9, 'min_child_weight': 2}

### DeepSurv
deepsurv_lung = {'batch_norm': True, 'dropout': 0.1, 'epochs': 27, 'reg': 0.1,
            'lr': 0.01, 'num_layers': 2, 'num_nodes': 32, 'output_bias': True}

deepsurv_veterans = {'batch_norm': True, 'dropout': 0.2, 'epochs': 21, 'reg': 0.1,
            'lr': 0.001, 'num_layers': 2, 'num_nodes': 16, 'output_bias': True}

deepsurv_udca = {'batch_norm': True, 'dropout': 0.1, 'epochs': 23, 'reg': 0.1,
            'lr': 0.01, 'num_layers': 2, 'num_nodes': 8, 'output_bias': True}

### DeepHit
deephit_lung = {'batch_norm': True, 'dropout': 0.2, 'epochs': 19, 'reg': 0.0001,
            'lr': 0.001, 'num_layers': 2, 'num_nodes': 32, 'output_bias': True}

deephit_veterans = {'batch_norm': True, 'dropout': 0.2, 'epochs': 25, 'reg': 0.0,
            'lr': 0.001, 'num_layers': 2, 'num_nodes': 16, 'output_bias': True}

deephit_udca = {'batch_norm': True, 'dropout': 0.3, 'epochs': 25, 'reg': 0.0001,
            'lr': 0.001, 'num_layers': 2, 'num_nodes': 8, 'output_bias': True}

def load_hyperparams(model : str, dataset : str):
    """
    Loads hyperparameters for a given model and dataset
    """
    if model=='rsf':
        if dataset=='lung':
            return rsf_lung
        elif dataset=='veterans':
            return rsf_veterans
        elif dataset=='udca':
            return rsf_udca
    elif model=='xgb':
        if dataset=='lung':
            return xgb_lung
        elif dataset=='veterans':
            return xgb_veterans
        elif dataset=='udca':
            return xgb_udca
    elif model=='deepsurv':
        if dataset=='lung':
            return deepsurv_lung
        elif dataset=='veterans':
            return deepsurv_veterans
        elif dataset=='udca':
            return deepsurv_udca
    elif model=='deephit':
        if dataset=='lung':
            return deephit_lung
        elif dataset=='veterans':
            return deephit_veterans
        elif dataset=='udca':
            return deephit_udca
