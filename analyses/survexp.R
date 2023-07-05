library(survival)
library(survex)

# Load the data
data <- read.csv("set1.csv")
delta_num <- as.numeric(data$delta == "True")
data$delta <- delta_num
head(data)
table(data$delta)
true_coef <- c(10**(-6), 0.1, -0.15, 10**(-6), 10**(-6))
new_observation <- data.frame(
  "one" = 0,
  "two" = 0,
  "three" = 0,
  "four" = 0,
  "five" = 0
)

# Fit a Cox model
cph <- survival::coxph(
  survival::Surv(time_to_event, delta) ~ one + two + three + four + five,
  data = data,
  model = TRUE,
  x = TRUE
)

cox_coefficients <- cph$coefficients

# Explain
num_rep <- 100
results <- matrix(nrow = num_rep, ncol = length(cox_coefficients) + 1)
for (i in (1:num_rep)) {
  init_time <- proc.time()
  exp <- explain(cph)
  p_parts_lime <- predict_parts(
    explainer = exp,
    new_observation = new_observation,
    N = 100,
    type = "survlime"
  )
  elapsed <- (proc.time() - init_time)[3]
  results[i, ] <- c(as.numeric(p_parts_lime$result), elapsed)
}

results <- as.data.frame(results)
colnames(results) <- c(colnames(data)[1:length(cox_coefficients)], "time")
write.csv(results, file = "survex_results.csv", row.names = FALSE)
