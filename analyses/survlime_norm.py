import pandas as pd
from survlimepy import SurvLimeExplainer
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
import time

data = pd.read_csv('set1.csv')
X = data.iloc[:, [0, 1, 2, 3 ,4]]
time_to_event = data.time_to_event
delta = data.delta
z = [(d, t) for d, t in zip(delta, time_to_event)]
y = np.array(z, dtype=[("delta", np.bool_), ("time_to_event", np.float32)])

# Fit a Cox model
cox = CoxPHSurvivalAnalysis()
cox.fit(X, y)
print(cox.coef_)

# Run the algorithm to obtain the coefficients
n_sim = 100
num_samples = 100
norm_to_use = 2
individual = [0, 0, 0, 0, 0]
B_individual = np.full(shape=(n_sim, len(individual) + 1), fill_value=np.nan)
for i in range(n_sim):
    print(f"Working on simulation {i}")
    start_time = time.time()
    explainer_cox = SurvLimeExplainer(
        training_features=X,
        training_events=delta,
        training_times=time_to_event,
        model_output_times=cox.event_times_,
        functional_norm=norm_to_use
    )
    b_cox = explainer_cox.explain_instance(
        data_row=individual,
        predict_fn=cox.predict_cumulative_hazard_function,
        num_samples=num_samples,
        verbose=False,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    B_individual[i, 0:len(b_cox)] = b_cox
    B_individual[i, -1] = elapsed_time

cols = list(data.columns)[:len(individual)]
cols.append("time")

df = pd.DataFrame(B_individual, columns=cols)
df.to_csv(f"survlime_{str(norm_to_use)}_results.csv", index=False)
