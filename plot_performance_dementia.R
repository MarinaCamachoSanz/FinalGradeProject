library(ggplot2)

# # # # # # # #
# SL Metrics # 
# # # # # # # #

AUC_SL <- c(0.75,0.75,0.58,0.74,0.75)
SD_AUC_SL <- c(0.02,0.01,0.01,0.02,0.02)

Precision_SL <- c(0.77,0.77,0.59,0.76,0.78)
SD_Precision_SL <- c(0.02,0.02,0.03,0.03,0.03)

Sensitivity_SL <- c(0.75,0.72,0.72,0.71,0.71)
SD_Sensitivity_SL <- c(0.02,0.02,0.07,0.02,0.02)

F1_SL <- c(0.74,0.74,0.58,0.73,0.74)
SD_F1_SL <- c(0.02,0.01,0.03,0.01,0.01)

# # # # # # # #
# ML Metrics # 
# # # # # # # #

AUC_ML <- c(0.83,0.77,0.73,0.82,0.77)
SD_AUC_ML <- c(0.02,0.01,0.02,0.02,0.01)

Precision_ML <- c(0.86,0.81,0.84,0.87,0.80)
SD_Precision_ML <- c(0.02,0.03,0.01,0.04,0.02)

Sensitivity_ML <- c(0.79,0.72,0.56,0.79,0.72)
SD_Sensitivity_ML <- c(0.01,0.02,0.03,0.02,0.02)

F1_ML <- c(0.82,0.76,0.67,0.83,0.76)
SD_F1_ML <- c(0.01,0.01,0.02,0.02,0.01)

Metric <- c('AUC','AUC','AUC','AUC','AUC','F1-score','F1-score','F1-score','F1-score','F1-score','Precision','Precision','Precision','Precision','Precision','Sensitivity','Sensitivity','Sensitivity','Sensitivity','Sensitivity','AUC','AUC','AUC','AUC','AUC','Precision','Precision','Precision','Precision','Precision','Sensitivity','Sensitivity','Sensitivity','Sensitivity','Sensitivity')
category <- c('Exposome with age','Exposome without age','Age','Accessible with age','Accessible without age')
values_metrics <- c(AUC_SL, F1_SL, Precision_SL, Sensitivity_SL, AUC_ML, F1_ML, Precision_ML, Sensitivity_ML)
values_sd <- c(SD_AUC_SL, SD_F1_SL, SD_Precision_SL, SD_Sensitivity_SL, SD_AUC_ML, SD_F1_ML, SD_Precision_ML, SD_Sensitivity_ML)
Metric_and_Model <- c(rep("AUC Statistical Learning", 5), rep("F1-score Statistical Learning", 5), rep("Precision Statistical Learning", 5), rep("Sensitivity Statistical Learning", 5), rep("AUC Machine Learning", 5), rep("F1-score Machine Learning", 5), rep("Precision Machine Learning", 5), rep("Sensitivity Machine Learning", 5))
Model <- c(rep("Logistic Regression", 20), rep("XGBoost", 20))
Metric <- c(rep("AUC", 5), rep("F1-score", 5), rep("Precision", 5), rep("Sensitivity", 5), rep("AUC", 5), rep("F1-score", 5), rep("Precision", 5), rep("Sensitivity", 5))
mydata <- data.frame(category, values_metrics, values_sd, Metric_and_Model, Model, Metric)

tiff(filename="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/Internal_Metrics_Plot_tim.tiff", width=2500, height=750, res=300)
pd <- position_dodge(.3)  # Save the dodge spec because we use it repeatedly
IV_Plot <- ggplot(data = mydata, mapping = aes(x=factor(category), y=values_metrics, colour=Model, group=Metric_and_Model))+
  geom_errorbar(aes(ymin=values_metrics-values_sd, ymax=values_metrics+values_sd),
                width = .2,
                size = 0.25,
                colour = "black",
                position = pd
  ) +
  geom_line(aes(linetype=Metric, color=Model),position = pd)+
  scale_linetype_manual(values=c("solid", "twodash", "dotted", "dashed"))+
  geom_point(position = pd, size = 2.5)+
  ggtitle('Internal Validation')+  
  theme_minimal()+
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank())+
  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
  scale_color_manual(values=c("#5FAC80", "#3F1151"))
IV_Plot
dev.off()

#################################################################
#################################################################
#EXTERNAL 

# # # # # # # #
# SL Metrics # 
# # # # # # # #

AUC_SL <- c(0.77,0.75,0.68,0.75,0.77)
SD_AUC_SL <- c(0.01,0.01,0.03,0.01,0.01)

Precision_SL <- c(0.86,0.85,0.68,0.84,0.84)
SD_Precision_SL <- c(0.01,0.01,0.04,0.01,0.01)

Sensitivity_SL <- c(0.64,0.64,0.69,0.63,0.67)
SD_Sensitivity_SL <- c(0.01,0.01,0.02,0.01,0.02)

F1_SL <- c(0.73,0.73,0.68,0.72,0.75)
SD_F1_SL <- c(0.01,0.01,0.01,0.01,0.01)

# # # # # # # #
# ML Metrics # 
# # # # # # # #

AUC_ML <- c(0.89,0.79,0.84,0.88,0.78)
SD_AUC_ML <- c(0.01,0.01,0.00,0.01,0.02)

Precision_ML <- c(0.93,0.96,0.94,0.87,0.86)
SD_Precision_ML <- c(0.01,0.02,0.00,0.01,0.03)

Sensitivity_ML <- c(0.83,0.68,0.71,0.81,0.67)
SD_Sensitivity_ML <- c(0.01,0.03,0.00,0.02,0.02)

F1_ML <- c(0.82,0.76,0.82,0.87,0.75)
SD_F1_ML <- c(0.01,0.01,0.00,0.01,0.03)

Metric <- c('AUC','AUC','AUC','AUC','AUC','F1-score','F1-score','F1-score','F1-score','F1-score','Precision','Precision','Precision','Precision','Precision','Sensitivity','Sensitivity','Sensitivity','Sensitivity','Sensitivity','AUC','AUC','AUC','AUC','AUC','Precision','Precision','Precision','Precision','Precision','Sensitivity','Sensitivity','Sensitivity','Sensitivity','Sensitivity')

category <- c('Exposome with age','Exposome without age','Age','Accessible with age','Accessible without age')
values_metrics <- c(AUC_SL, F1_SL, Precision_SL, Sensitivity_SL, AUC_ML, F1_ML, Precision_ML, Sensitivity_ML)
values_sd <- c(SD_AUC_SL, SD_F1_SL, SD_Precision_SL, SD_Sensitivity_SL, SD_AUC_ML, SD_F1_ML, SD_Precision_ML, SD_Sensitivity_ML)
Metric_and_Model <- c(rep("AUC Statistical Learning", 5), rep("F1-score Statistical Learning", 5), rep("Precision Statistical Learning", 5), rep("Sensitivity Statistical Learning", 5), rep("AUC Machine Learning", 5), rep("F1-score Machine Learning", 5), rep("Precision Machine Learning", 5), rep("Sensitivity Machine Learning", 5))
Model <- c(rep("Logistic Regression", 20), rep("XGBoost", 20))
Metric <- c(rep("AUC", 5), rep("F1-score", 5), rep("Precision", 5), rep("Sensitivity", 5), rep("AUC", 5), rep("F1-score", 5), rep("Precision", 5), rep("Sensitivity", 5))
mydata <- data.frame(category, values_metrics, values_sd, Metric_and_Model, Model, Metric)

tiff(filename="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/External_Metrics_Plot_tim.tiff", width=2500, height=750, res=300)
pd <- position_dodge(.3)  # Save the dodge spec because we use it repeatedly
IV_Plot <- ggplot(data = mydata, mapping = aes(x=factor(category), y=values_metrics, colour=Model, group=Metric_and_Model))+
  geom_errorbar(aes(ymin=values_metrics-values_sd, ymax=values_metrics+values_sd),
                width = .2,
                size = 0.25,
                colour = "black",
                position = pd
  ) +
  geom_line(aes(linetype=Metric, color=Model),position = pd)+
  scale_linetype_manual(values=c("solid", "twodash", "dotted", "dashed"))+
  geom_point(position = pd, size = 2.5)+
  ggtitle('External Validation')+  
  theme_minimal()+
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank())+
  scale_x_discrete(guide = guide_axis(n.dodge = 2))+
  scale_color_manual(values=c("#5FAC80", "#3F1151"))
IV_Plot
dev.off()
