import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split


def prepare_splits(x : pd.DataFrame, y : np.ndarray, 
                        split_sizes : list = [0.1, 0.5], random_state : int=1) -> [list, list]:
  """
  Creates a random split with given sizes and random seed.
  
  Parameters:
  x : pd.DataFrame: variable containing the dataset
  y : np.ndarray: variable containing the target values
  split_sizes: list: variable containing the percentage of the first and second split
  random_state: int: seed of the random state to be used when splitting
  
  Returns:
  X_x: pd.DataFrame: three different DataFrames with the splits' data
  y_y: ----: three different ---- with the splits' targets
  """
  X_train, X_test, y_train, y_test = train_test_split(x.copy(), y, test_size=split_size[0], random_state=random_state)
  X_val, X_test, y_val, y_test = train_test_split(X_test.copy(), y_test, test_size=split_size[1], random_state=random_state)
  scaler = StandardScaler()

  X_train_processed = pd.DataFrame(data=scaler.fit_transform(X_train, y_train),
                                   columns=X_train.columns, index=X_train.index)

  X_test_processed = pd.DataFrame(data=scaler.transform(X_test),
                                   columns=X_test.columns, index=X_test.index)

  #print(hasattr(scaler, "n_features_in_"))
  events_train = [x[0] for x in y_train]
  times_train  = [x[1] for x in y_train]

  events_val = [x[0] for x in y_val]
  times_val  = [x[1] for x in y_val]

  events_test = [x[0] for x in y_test]
  times_test  = [x[1] for x in y_test]

  y_train = Surv.from_arrays(events_train, times_train)
  y_val   = Surv.from_arrays(events_val, times_val)
  y_test  = Surv.from_arrays(events_test, times_test)
  
  return X_train_processed, y_train, X_val, y_val, X_test_processed, y_test