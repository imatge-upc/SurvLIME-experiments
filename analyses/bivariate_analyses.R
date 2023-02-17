# Preliminary ----
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
install.packages(setdiff("tidyverse", rownames(installed.packages())))
library("tidyverse")

# UDCA ----
data <- read.csv("udca_dataset.csv")
head(data)
data_filtered <- data %>% 
  drop_na()
data_filtered$bili_cat <- cut_number(data_filtered$bili, 4)
data_filtered$riskscore_cat <- cut_number(data_filtered$riskscore, 4)
variables <- c("trt", "stage", "bili_cat", "riskscore_cat")
for (var in variables) {
  results <- data_filtered %>%
    group_by(across(all_of(var))) %>% 
    summarise(total=n(), event=sum(status), percentage_event = 100*sum(status)/n())  
  print(results)
  print("-----------")
}

# Percentage of individuals that experience the event
data %>%
  drop_na() %>%
  select(status) %>% 
  summarise(mean(status))

# LUNG ----
data <- read.csv("lung_dataset.csv")
data$status <- data$status - 1
unique(data$inst)
head(data)
data_filtered <- data %>% 
  select(-c(meal.cal, inst)) %>% 
  drop_na() %>% 
  filter(ph.ecog != "3")
data_filtered$age_cat <- cut_number(data_filtered$age, 4)
data_filtered$ph_karno_cat <- cut_number(data_filtered$ph.karno, 3)
data_filtered$pat_karno_cat <- cut_number(data_filtered$pat.karno, 4)
data_filtered$wt_loss_cat <- cut_number(data_filtered$wt.loss, 4)
variables <- c("sex", "ph.ecog", "age_cat", "ph_karno_cat", "pat_karno_cat", "wt_loss_cat")
for (var in variables) {
  results <- data_filtered %>%
    group_by(across(all_of(var))) %>% 
    summarise(total=n(), event=sum(status), percentage_event = 100*sum(status)/n())
  print(results)
  print("-----------")
}

# Percentage of individuals that experience the event
data %>%
  select(-c(meal.cal, inst, meal_cal_cat)) %>% 
  drop_na() %>% 
  filter(ph.ecog != "3") %>% 
  select(status) %>% 
  summarise(mean(status))

# Veteran ----
data <- read.csv("veteran.csv")
data$celltype <- factor(data$celltype, levels = c("squamous", "smallcell", "adeno", "large"))
head(data)
data$karno_cat <- cut_number(data$karno, 4)
data$age_cat <- cut_number(data$age, 4)
data$diagtime_cat <- cut_number(data$diagtime, 4)
variables <- c("trt", "prior", "celltype", "karno_cat", "age_cat", "diagtime_cat")
for (var in variables) {
  results <- data %>%
    group_by(across(all_of(var))) %>% 
    summarise(total=n(), event=sum(status), percentage_event = 100*sum(status)/n())
  print(results)
  print("-----------")
}

# Percentage of individuals that experience the event
data %>%
  summarise(mean(status))
