library(ggplot2)

#'AUC','F1-score','Precission','Sensitivity'
#LR <- c(0.80,0.79,0.82,0.77)
#LR <- c(0.80,0.79,0.82,0.77)

#LR_sd <- c(0.02,0.02,0.04,0.02)
#LR_sd_ex <- c(0.01,0.01,0.01,0.01)

AUC_SL <- c(0.80,0.80)
SD_AUC_SL <- c(0.02,0.01)

Precision_SL <- c(0.82,0.82)
SD_Precision_SL <- c(0.04,0.01)

Sensitivity_SL <- c(0.77,0.77)
SD_Sensitivity_SL <- c(0.02,0.01)

F1_SL <- c(0.79,0.79)
SD_F1_SL <- c(0.02,0.01)

#'AUC','F1-score','Precission','Sensitivity'
#XGB <- c(0.90,0.89,0.96,0.83)
#XGB <- c(0.89,0.88,0.92,0.85)

#XGB_sd <- c(0.02,0.03,0.02,0.05)
#XGB_sd_ex <- c(0.01,0.01,0.02,0.01)

AUC_ML <- c(0.90,0.89)
SD_AUC_ML <- c(0.02,0.01)

Precision_ML <- c(0.96,0.92)
SD_Precision_ML <- c(0.02,0.02)

Sensitivity_ML <- c(0.83,0.83)
SD_Sensitivity_ML <- c(0.05,0.01)

F1_ML <- c(0.89,0.89)
SD_F1_ML <- c(0.03,0.01)

category <- c('AUC','AUC','F1-score','F1-score','Precision','Precision','Sensitivity','Sensitivity','AUC','AUC','F1-score','F1-score','Precision','Precision','Sensitivity','Sensitivity')
values_metrics <- c(AUC_SL, F1_SL, Precision_SL, Sensitivity_SL, AUC_ML, F1_ML, Precision_ML, Sensitivity_ML)
values_sd <- c(SD_AUC_SL, SD_F1_SL, SD_Precision_SL, SD_Sensitivity_SL, SD_AUC_ML, SD_F1_ML, SD_Precision_ML, SD_Sensitivity_ML)
Metric_and_Model <- c('Internal Validation Logistic Regression', 'External Validation Logistic Regression','Internal Validation Logistic Regression', 'External Validation Logistic Regression','Internal Validation Logistic Regression', 'External Validation Logistic Regression','Internal Validation Logistic Regression', 'External Validation Logistic Regression','Internal Validation XGBoost', 'External Validation XGBoost','Internal Validation XGBoost', 'External Validation XGBoost','Internal Validation XGBoost', 'External Validation XGBoost','Internal Validation XGBoost', 'External Validation XGBoost')
Model <- c(rep("Logistic Regression", 8), rep("XGBoost", 8))
Validation <- c('Internal','External','Internal','External','Internal','External','Internal','External','Internal','External','Internal','External','Internal','External','Internal','External')
mydata <- data.frame(category, values_metrics, values_sd, Metric_and_Model, Model, Validation)

tiff(filename="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/PlotTFG_DvsDD.tiff", width=2500, height=750, res=300)
pd <- position_dodge(.3)  # Save the dodge spec because we use it repeatedly
IV_Plot <- ggplot(data = mydata, mapping = aes(x=factor(category), y=values_metrics, colour=Model, group=Metric_and_Model))+
  geom_errorbar(aes(ymin=values_metrics-values_sd, ymax=values_metrics+values_sd),
                width = .2,
                size = 0.25,
                colour = "black",
                position = pd
  ) +
  geom_point(position = pd, size = 2.5)+
  geom_line(aes(linetype=Validation, color=Model),position = pd)+
  scale_linetype_manual(values=c("solid", "dashed"))+
  ggtitle('Performance of comorbid dementia and depression model')+
  theme_minimal()+
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank())+
  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
  scale_color_manual(values=c("#AAD55F", "#426D8B"))
IV_Plot
dev.off()
