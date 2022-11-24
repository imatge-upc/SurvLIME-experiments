from survLime.survshap import SurvivalModelExplainer, PredictSurvSHAP, ModelSurvSHAP
import pickle

import warnings
warnings.filterwarnings("ignore")


import sklearn
from sksurv.ensemble import RandomSurvivalForest

from sklearn.model_selection import train_test_split
import sklearn


from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import seaborn as sns
sns.set()


from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split


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

from derma.general.preprocessing.encoders import (OrdinalEncoder, #CustomOrdinalEncoder as 
                                                  GenderEncoder,
                                                  AbsentPresentEncoder,
                                                  LABEncoder,
                                                  CategoricalEncoder)

from derma.general.ingestion.data_loader_csv import SurvivalLoader


import config_os as settings_file

np.random.seed(123456)

get_target = lambda df: df[['event','duration']]
path = '/home/carlos.hernandez/datasets/csvs/data-surv_20220302.csv'
X, _, time, event = SurvivalLoader('ss').load_data(path)
X['duration'] = round(time)

X['event']    = event
X.dropna(subset=['duration'],inplace=True)
X = X[X['duration']>=0]
X_train_previous, X_test, y_train, y_test = train_test_split(X.copy(), np.zeros(len(X)), test_size=0.1, random_state=42)
kwargs = {'n_estimators' : 300,
                    'min_weight_fraction_leaf': 0,
                    'min_samples_split' : 11,
                    'max_features' : 'auto',
                    'max_depth':10}
model_pipe = ('rsf', RandomSurvivalForest(
                                  **kwargs,
                                  random_state=42))
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
            model_pipe
            ])



X_train, X_val, y_train, y_val = train_test_split(X_train_previous.copy(),
        np.zeros(len(X_train_previous)), test_size=0.2, random_state=0)
y_train_list = get_target(X_train)
y_test = get_target(X_test)
y_train_list = Surv.from_dataframe(*y_train_list.columns, y_train_list)
y_test = Surv.from_dataframe(*y_test, y_test)

#import ipdb;ipdb.set_trace() 
fitted_pipe = pipe.fit(X_train.copy(), y_train_list.copy())

X_test_t = fitted_pipe[:-1].transform(X_test)
X_train_t = fitted_pipe[:-1].transform(X_train)    
smaller_model = pipe[-1].fit(X_train_t.iloc[:,:10], y_train_list)
    
rsf_exp = SurvivalModelExplainer(smaller_model[-1], X_test_t.iloc[:, :10], y_test)
exp1_survshap_global_rsf = ModelSurvSHAP(random_state=42)
exp1_survshap_global_rsf.fit(rsf_exp)

with open("pickles/xxmm_survshap_ss_event", "wb") as file:
    pickle.dump(exp1_survshap_global_rsf, file)
