import sklearn
import numpy as np
import pandas as pd
from typing import Tuple

import wandb
from torch.utils.data import DataLoader 
import torchtuples as tt
import torchvision
from torchvision import transforms
from torchtuples.callbacks import Callback
from pycox.evaluation import EvalSurv
from sklearn.model_selection import StratifiedKFold, train_test_split

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
                                                      FillWith0Transformer,
                                                      CustomImputer)
from derma.general.preprocessing.encoders import (OrdinalEncoder,
                                                  GenderEncoder,
                                                  AbsentPresentEncoder,
                                                  LABEncoder,
                                                  CategoricalEncoder)
import derma.sol.survival.notebooks.config_os as settings_file
import derma.sol.survival.notebooks.config_os_marc as settings_file


## Xgboost imports
import xgboost as xgb
from xgbse.metrics import concordance_index
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)


path = '/home/carlos.hernandez/datasets/csvs/data-surv_20220302.csv'

# Create a function that does nested cross-validation
def nested_stratified_split(X, n_splits=4, inner_splits=3, random_state=314):
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    splits_outer= []
    for train_idx, test_idx in kf.split(X, X['event']):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # create a kfold on X_train to create an X_val set
        inner_kf = StratifiedKFold(n_splits=inner_splits, random_state=random_state, shuffle=True)
        splits_inner = []
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train, X_train['event']):
            X_train_inner = X_train.iloc[inner_train_idx]
            X_val = X_train.iloc[inner_val_idx]
            splits_inner.append([X_train_inner, X_val])
        splits_outer.append([splits_inner, X_train, X_test])
    return splits_outer

def data_preprocessing(X_train : pd.DataFrame, X_val : pd.DataFrame, X_test : pd.DataFrame,
                       using_dataloaders: False, wsi_dict={'train' : []}, dermo_dict={'train':[]}) -> Tuple:
    """
    Applies all the necessary transformation for SKSurv/XGBSE or PyCox to be used
    """
    pipe = obtain_pipeline(settings_file, X_train)
    # Transform the data so we can use it
    X_train_processed = pipe.fit_transform(X_train.copy(), np.zeros(len(X_train)))
    X_val_processed   = pipe.transform(X_val.copy())
    if X_test is not None:
        X_test_processed   = pipe.transform(X_test.copy())
    else: 
        X_test_processed = None
    columns = X_train_processed.columns
    if not using_dataloaders:
        X_train_processed = X_train_processed.to_numpy().astype('float32')
        X_val_processed   = X_val_processed.to_numpy().astype('float32')
        if X_test is not None:
            X_test_processed  = X_test_processed.to_numpy().astype('float32')
    elif len(wsi_dict['train'])>0:
        X_train_processed['slide_id'] = wsi_dict['train']
        X_val_processed['slide_id'] = wsi_dict['val']
        if X_test is not None:
            X_test_processed['slide_id'] = wsi_dict['test']
    if using_dataloaders and len(dermo_dict['train'])>0:
        X_train_processed['filename'] = dermo_dict['train']
        X_val_processed['filename'] = dermo_dict['val']
        if X_test is not None:
            X_test_processed['filename'] = dermo_dict['test']
    
    return X_train_processed, X_val_processed, X_test_processed, columns

def obtain_pipe(keep_cols : list = None, impute : bool = True):
    """
    Give the settings of the pipeline instantiates the sk Pipeline to process the data

    settings : dict : Should be given when loading the data from the extracted SQL query
    """
    if keep_cols is not None:
        settings_file.keep_cols = {'cols' : keep_cols}
    pipe = Pipeline(steps=[
        ('TransformToNumeric', TransformToNumeric(**settings_file.transform_to_numeric)), 
        ('TransformToDatetime', TransformToDatetime(**settings_file.transform_to_datetime)),
        ('TransformToObject', TransformToObject(**settings_file.transform_to_object)),
        ('FillWith0', FillWith0Transformer(**settings_file.fill_with_0)),
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
        ('TransformNodMets', TransformNodMets(**settings_file.transform_nod_mets)),
        ('KeepColumns', KeepColumns(**settings_file.keep_cols)),
        ('RenameLabValues', RenameLabValues(**settings_file.rename_lab_values)),
       # ('CustomImputer', CustomImputer(strategy='mean')),
        ('CustomScaler', CustomScaler()),
#        ('clf', xgb.XGBRegressor(**vars(args)))
        ])
    # if impute is true add CustomScaler to pipeline
    if impute:
        pipe.steps.insert(-1, ('CustomImputer', CustomImputer(strategy='mean')))
    return pipe


class Concordance(tt.cb.MonitorMetrics):
    def __init__(self, x, durations, events, per_epoch=5, verbose=True, wandb_metric : str='none', use_wandb : bool=False):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose
        self.wandb_metric   = wandb_metric 
        self.use_wandb = use_wandb
    
    def on_epoch_end(self):
        super().on_epoch_end()
        # log in wandb self.model_val_metrics.scores['loss']['score'][-1]
        if self.use_wandb:
            wandb.log({'Validation loss' : self.model.val_metrics.scores['loss']['score'][-1],
                      'Train loss' : self.model.train_metrics.scores['loss']['score'][-1]})
        if self.epoch % self.per_epoch == 0:
            self.model.net.eval()
            surv = self.model.interpolate(20).predict_surv_df(self.x)
            self.model.net.train()
            ev = EvalSurv(surv, self.durations, self.events)

            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            
            if self.verbose:
                print(f'C-index val', concordance)
                if self.use_wandb:
                    wandb.log({'C-index val': concordance}, commit=False)
    
    def get_last_score(self):
        return self.scores['concordance']['score'][-1]
                
def obtain_label_funct(args):
    if args.ml:
        label_funct = lambda df: df[['event','duration']]
    else:
        label_funct = lambda df: (df['duration'].values, df['event'].values)
    return label_funct 



def obtain_pipeline(settings_file : dict, X_train : pd.DataFrame) -> sklearn.pipeline.Pipeline:
    """
    Give the settings of the pipeline instantiates the sk Pipeline to process the data

    settings : dict : Should be given when loading the data from the extracted SQL query
    """
    
   ## obtain values of keep columns from settings_file 
   #keep_columns = list(settings_file.keep_cols.values())[0]
   ## delete from keep_columns all the columns from X_train with all missing data
   #keep_columns = [col for col in keep_columns if X_train[col].isnull().sum() < X_train.shape[0]]
   #settings_file.keep_columns = {'cols' : keep_columns}
    pipe = sklearn.pipeline.Pipeline(steps=[
        ('TransformToNumeric', TransformToNumeric(**settings_file.transform_to_numeric)), 
        ('TransformToDatetime', TransformToDatetime(**settings_file.transform_to_datetime)),
        ('TransformToObject', TransformToObject(**settings_file.transform_to_object)),
        ('FillWith0', FillWith0Transformer(**settings_file.fill_with_0)),
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
        ('CustomImputer', CustomImputer(strategy='mean')),
        ('CustomScaler', CustomScaler()),
#       ('RenameLabValues', RenameLabValues(**settings_file.rename_lab_values)),
        ])

    return pipe



def use_xgboost(data : list, labels : list, args : float) -> float:
    """Use XGBoost to predict survival"""

    y_train = convert_to_structured(labels[0]['duration'], labels[0]['event'])
    y_val = convert_to_structured(labels[1]['duration'], labels[1]['event'])
    y_test = convert_to_structured(labels[2]['duration'], labels[2]['event'])
    PARAMS_XGB_AFT = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': args.aft_loss_distribution,
        'aft_loss_distribution_scale': args.aft_loss_distribution_scale,
        'tree_method': 'hist', 
        'learning_rate': args.lr, 
        'max_depth': args.max_depth,
        'alpha': args.alpha,
        'booster':'dart',
        'subsample':0.5,
        'min_child_weight': args.min_child_weight,
        'colsample_bynode':0.5
    }
    # converting to xgboost format
    dtrain = convert_data_to_xgb_format(data[0], y_train, 'survival:aft')
    dval = convert_data_to_xgb_format(data[1], y_val, 'survival:aft')

    bst = xgb.train(
            PARAMS_XGB_AFT, dtrain,
            num_boost_round=200,
             verbose_eval=0
        )
    dtest = convert_data_to_xgb_format(data[2], y_test, 'survival:aft')

    preds = bst.predict(dtest)
    score = concordance_index(y_test, -preds, risk_strategy='precomputed')
    return score
