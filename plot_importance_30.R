setwd("/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG")
SL_30_with_age <- read.csv("FeatImport_LR0_7_age.csv", sep = ";")
ML_30_with_age <- read.csv("FeatImport_XG0_7_age.csv", sep = ";")
SL_30_without_age <- read.csv("FeatImport_LR0_7.csv", sep = ";")
ML_30_without_age <- read.csv("FeatImport_XG0_7.csv", sep = ";")

tiff(filename="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/SL_30_with_age_tim.tiff", width=2700, height=2000, res=300)
p_SL_30_with_age <- ggplot(SL_30_with_age, aes(x=feature_names,y=feature_importance))+
  geom_col(aes(fill = feature_importance))+
  #scale_fill_viridis(option = "rainbow") +
  scale_fill_continuous(type = "viridis") +
  coord_flip() +
  theme_minimal() +
  theme(legend.position=c(0.95,0.8)) +
  theme(legend.title=element_blank()) +
  ggtitle('Accessible with age experiment: Statistical learning')+
  theme(plot.title=element_text(hjust = 0.5))+
  xlab("Exposure") + ylab("Exposure's importance")
p_SL_30_with_age
dev.off()

tiff(filename="/Users/marinacamacho/Desktop/BCN-AIM/DementiaTFG/ML_30_with_age_tim.tiff", width=2700, height=2000, res=300)
p_ML_30_with_age <- ggplot(ML_30_with_age, aes(x=feature_names,y=feature_importance))+
  geom_col(aes(fill = feature_importance))+
  scale_fill_continuous(type = "viridis") +
  coord_flip() +
  theme_minimal() +
  theme(legend.position=c(0.95,0.2)) +
  theme(legend.title=element_blank()) +
  ggtitle('Accessible with age experiment: Machine learning')+
  theme(plot.title=element_text(hjust = 0.5))+
  xlab("Exposure") + ylab("Exposure's importance")
p_ML_30_with_age
dev.off()






