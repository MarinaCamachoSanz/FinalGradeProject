setwd("/Users/marinacamachosanz/Desktop/tfg_BCN-AIM/final")
set.seed(27)

#input_nan_df_c <- read.csv("input_nan_df_c.csv")
#input_nan_df_t <- read.csv("input_nan_df_t.csv")
#exval_nan_df_c <- read.csv("exval_nan_df_c.csv")
#exval_nan_df_t <- read.csv("exval_nan_df_t.csv")

input_nan_df_c_80 <- read.csv("input_nan_df_c_80.csv")
input_nan_df_t_80 <- read.csv("input_nan_df_t_80.csv")
exval_nan_df_c_80 <- read.csv("exval_nan_df_c_80.csv")
exval_nan_df_t_80 <- read.csv("exval_nan_df_t_80.csv")

#input_df_c <- missForest(input_nan_df_c)$ximp
#input_df_t <- missForest(input_nan_df_t)$ximp
#exval_df_c <- missForest(exval_nan_df_c)$ximp
#exval_df_t <- missForest(exval_nan_df_t)$ximp

input_df_c_80 <- missForest(input_nan_df_c_80)$ximp
input_df_t_80 <- missForest(input_nan_df_t_80)$ximp
exval_df_c_80 <- missForest(exval_nan_df_c_80)$ximp
exval_df_t_80 <- missForest(exval_nan_df_t_80)$ximp

#sum(is.na(input_nan_df_c))
#sum(is.na(input_nan_df_t))
#sum(is.na(exval_nan_df_c))
#sum(is.na(exval_nan_df_t))
#sum(is.na(input_df_c))
#sum(is.na(input_df_t))
#sum(is.na(exval_df_c))
#sum(is.na(exval_df_t))

sum(is.na(input_nan_df_c_80))
sum(is.na(input_nan_df_t_80))
sum(is.na(exval_nan_df_c_80))
sum(is.na(exval_nan_df_t_80))
sum(is.na(input_df_c_80))
sum(is.na(input_df_t_80))
sum(is.na(exval_df_c_80))
sum(is.na(exval_df_t_80))

#write.csv(input_df_c,"input_df_c.csv", row.names = TRUE)
#write.csv(input_df_t,"input_df_t.csv", row.names = TRUE)
#write.csv(exval_df_c,"exval_df_c.csv", row.names = TRUE)
#write.csv(exval_df_t,"exval_df_t.csv", row.names = TRUE)

write.csv(input_df_c_80,"input_df_c_80.csv", row.names = TRUE)
write.csv(input_df_t_80,"input_df_t_80.csv", row.names = TRUE)

write.csv(exval_df_c_80,"exval_df_c_80.csv", row.names = TRUE)
write.csv(exval_df_t_80,"exval_df_t_80.csv", row.names = TRUE)

