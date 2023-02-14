# Preliminary ----
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
if (!require(require(tidyverse))) {
  install.packages("tidyverse")
} else {
  library("tidyverse")
}

# UDCA ----
data <- read.csv("udca_dataset.csv")
head(data)
data$bili_cat <- cut_number(data$bili, 4)
data$riskscore_cat <- cut_number(data$riskscore, 4)
variables <- c("trt", "stage", "bili_cat", "riskscore_cat")
for (var in variables) {
  results <- data %>%
    drop_na() %>% 
    group_by(across(all_of(var))) %>% 
    summarise(total=n(), event=sum(status), percentage_event = 100*sum(status)/n())  
  print(results)
  print("-----------")
}

# LUNG ----
data <- read.csv("lung_dataset.csv")
data$status <- data$status - 1
unique(data$inst)
head(data)
data$age_cat <- cut_number(data$age, 4)
data$ph_karno_cat <- cut_number(data$ph.karno, 4)
data$pat_karno_cat <- cut_number(data$pat.karno, 4)
data$meal_cal_cat <- cut_number(data$meal.cal, 4)
data$wt_loss_cat <- cut_number(data$wt.loss, 4)
variables <- c("sex", "ph.ecog", "age_cat", "ph_karno_cat", "pat_karno_cat", "wt_loss_cat")
for (var in variables) {
  results <- data %>%
    select(-c(meal.cal, inst)) %>% 
    drop_na() %>% 
    filter(ph.ecog != "3") %>% 
    group_by(across(all_of(var))) %>% 
    summarise(total=n(), event=sum(status), percentage_event = 100*sum(status)/n())
  print(results)
  print("-----------")
}

# Veteran ----
data <- read.csv("veteran.csv")
data$celltype <- factor(data$celltype, levels = c("squamous", "smallcell", "adeno", "large"))
head(data)
data$karno_cat <- cut_number(data$karno, 4)
data$age_cat <- cut_number(data$age, 4)
variables <- c("trt", "prior", "celltype", "karno_cat", "age_cat")
for (var in variables) {
  results <- data %>%
    group_by(across(all_of(var))) %>% 
    summarise(total=n(), event=sum(status), percentage_event = 100*sum(status)/n())
  print(results)
  print("-----------")
}
