
library(tidyverse)
library(GGally)
library(corrplot)

#choosing file
df= read.csv(choose.files())
head(df)

#EDA
df$Crop <- trimws(df$Crop)
df$Season <- trimws(df$Season)
df$State <- trimws(df$State)

summary(df)
colSums(is.na(df))

ggplot(df,aes(Yield))+
  geom_histogram(bins=10)

#CROP vs yield
df %>%
  group_by(Crop) %>%
  summarise(mean_yield=mean(Yield)) %>%
  arrange(desc(mean_yield)) %>%
  head(20) %>%
  ggplot(aes(reorder(Crop,mean_yield),mean_yield))+
  geom_col()+
  coord_flip()

#state vs yield 
df %>%
  group_by(State) %>%
  summarise(mean_yield=mean(Yield)) %>%
  ggplot(aes(reorder(State,mean_yield),mean_yield))+
  geom_col()+
  coord_flip()

#CORRELATION ANALYSIS 
num <- df %>%
  select(
    Crop_Year,
    Area,
    Production,
    Annual_Rainfall,
    Fertilizer,
    Pesticide,
    Yield
  )

corrplot(cor(num))

cor(df[,c("Area","Fertilizer","Pesticide","Production","Yield")])
library(corrplot)

corrplot(
  cor(df[,c("Area","Fertilizer","Pesticide","Production","Yield")]),
  method = "number"
)

#MULTICOLLINEARITY CHECK
library(car)

lm1 <- lm(
  Yield ~ Area + Production +
    Fertilizer + Pesticide +
    Annual_Rainfall,
  data=df
)
vif(lm1)

#ANOVA 
crop_anova <- aov(
  Yield ~ Crop,
  data=df
)
summary(crop_anova)

state_anova <- aov(
  Yield ~ State,
  data=df
)
summary(state_anova)

season_anova <- aov(Yield ~ Season, data=df)
summary(season_anova)

twoway <- aov(
  Yield ~ Crop + State+ Season,
  data=df
)

summary(twoway)

tuk=TukeyHSD(crop_anova)
head(tuk)
head(tuk$Crop)
plot(tuk)



#CLUSTER ANALYSIS
cluster_df <- df %>%
  group_by(Crop) %>%
  summarise(
    Area=mean(Area),
    Rainfall=mean(Annual_Rainfall),
    Yield=mean(Yield)
  )

library(factoextra)
scaled <- scale(cluster_df[,2:4])

fviz_nbclust(
  scaled,
  kmeans,
  method="wss"
)
#NOT TRANSFORMED 
#K MEANS
set.seed(123)
km <- kmeans(
  scaled,
  centers=4,
  nstart=25
)
cluster_df$Cluster <- factor(km$cluster)
fviz_cluster(
  km,
  data=scaled
)
cluster_df$Cluster <- km$cluster
table(km$cluster)

aggregate(cluster_df[,2:ncol(cluster_df)-1],
          by=list(cluster_df$Cluster),
          mean)
cluster_df$Cluster <- factor(km$cluster)
cluster_df$Cluster <- km$cluster
split(cluster_df$Crop, cluster_df$Cluster)

#TRANSFORMED 
cluster_df %>%
  arrange(Cluster) %>%
  select(Crop, Cluster)

cluster_df$Area <- log1p(cluster_df$Area)
cluster_df$Yield <- log1p(cluster_df$Yield)

scaled <- scale(cluster_df[,2:4])

km <- kmeans(
  scaled,
  centers = 4,
  nstart = 50
)
fviz_cluster(
  km,
  data=scaled
)
cluster_df$Cluster <- km$cluster
table(km$cluster)

aggregate(cluster_df[,2:ncol(cluster_df)-1],
          by=list(cluster_df$Cluster),
          mean)

cluster_df$Cluster <- km$cluster
split(cluster_df$Crop, cluster_df$Cluster)

#HIERARCHIAL CLUSTERING
scaled_data <- scale(cluster_df[,2:4])
d <- dist(scaled_data)
hc <- hclust(d)
plot(hc)
rect.hclust(hc,
            k = 3,
            border = "red")

aggregate(
  cluster_df[,2:4],
  by=list(cluster_df$Cluster),
  mean
)



library(dplyr)
cluster_df %>%
  group_by(Cluster) %>%
  count(Crop) %>%
  arrange(Cluster, desc(n))

cluster_df %>%
  group_by(Cluster, Crop) %>%
  summarise(n = n(), .groups = "drop") %>%
  arrange(Cluster, desc(n)) %>%
  group_by(Cluster) %>%
  slice_head(n = 5)

cluster_df %>%
  group_by(Cluster, State) %>%
  summarise(n = n(), .groups = "drop") %>%
  arrange(Cluster, desc(n)) %>%
  group_by(Cluster) %>%
  slice_head(n = 5)

#MODEL FOR YIELD
#LINEAR REGRESSION USING LOG TRANSFORM
library(caret)
set.seed(123)
df$logYield <- log1p(df$Yield)

idx <- createDataPartition(
  df$logYield,
  p = 0.8,
  list = FALSE
)

train <- df[idx, ]
test  <- df[-idx, ]

lm_model <- lm(
  logYield ~ Crop +
    Season +
    State +
    Crop_Year +
    Annual_Rainfall,
  data = train
)

pred_lm <- predict(
  lm_model,
  newdata = test
)

RMSE(pred_lm, test$logYield)
R2(pred_lm, test$logYield)

#RANDOM FOREST 
library(randomForest)
rf_model <- randomForest(
  Yield ~ . - Production,
  data=train,
  ntree=500
)

pred_rf <- predict(
  rf_model,
  test
)

RMSE(pred_rf,test$Yield)
R2(pred_rf,test$Yield)

#log transform
rf_model <- randomForest(
  logYield ~ . - Yield - Production,
  data = train,
  ntree = 500,
  importance = TRUE
)

pred_rf <- predict(
  rf_model,
  newdata = test
)

RMSE(pred_rf, test$logYield)
R2(pred_rf, test$logYield)

#XGBOOST
library(caret)
library(xgboost)

dummy <- dummyVars(
  logYield~.-Production,
  data=df
)
X <- predict(
  dummy,
  df
)
y <- log1p(df$Yield)

x_train <- X[idx,]
x_test <- X[-idx,]

y_train <- y[idx]
y_test <- y[-idx]

xgb_model <- xgboost(
  data=as.matrix(x_train),
  label=y_train,
  objective="reg:squarederror",
  nrounds=300,
  max_depth=6,
  eta=.05,
  verbose=0
)

pred_xgb <- predict(
  xgb_model,
  as.matrix(x_test)
)

RMSE(pred_xgb,y_test)
R2(pred_xgb,test$logYield)


#RIDGE REGRESSION
library(caret)
library(glmnet)

# Train-test split
set.seed(123)

idx <- createDataPartition(
  df$logYield,
  p = 0.8,
  list = FALSE
)

train <- df[idx, ]
test  <- df[-idx, ]

# Design matrices
X_train <- model.matrix(
  logYield ~ Crop + State + Season +
    Crop_Year + Area +
    Fertilizer + Pesticide +
    Annual_Rainfall,
  data = train
)[,-1]

X_test <- model.matrix(
  logYield ~ Crop + State + Season +
    Crop_Year + Area +
    Fertilizer + Pesticide +
    Annual_Rainfall,
  data = test
)[,-1]

y_train <- train$logYield
y_test  <- test$logYield

# Ridge Regression
ridge <- cv.glmnet(
  X_train,
  y_train,
  alpha = 0
)

# Predictions
pred_ridge <- as.numeric(
  predict(
    ridge,
    newx = X_test,
    s = "lambda.min"
  )
)

# Performance
ridge_rmse <- RMSE(pred_ridge, y_test)
ridge_r2 <- R2(pred_ridge, y_test)

cat("Lambda =", ridge$lambda.min, "\n")
cat("RMSE =", ridge_rmse, "\n")
cat("R² =", ridge_r2, "\n")

coef_df <- data.frame(
  Variable = rownames(coef(ridge, s = "lambda.min")),
  Coefficient = as.numeric(
    coef(ridge, s = "lambda.min")
  )
)

compact_model <- subset(
  coef_df,
  abs(Coefficient) > 0.1
)

compact_model[
  order(abs(compact_model$Coefficient),
        decreasing = TRUE),
]

subset(
  coef_df,
  grepl(
    "Area|Annual_Rainfall|Fertilizer|Pesticide|Crop_Year",
    Variable
  )
)
plot(
  y_test,
  pred_ridge,
  pch = 16,
  xlab = "Actual log(Yield)",
  ylab = "Predicted log(Yield)",
  main = "Ridge Regression"
)

abline(0, 1, lwd = 2)


#LASSO
library(glmnet)

X <- model.matrix(
  Yield ~ Crop + State + Season +
    Crop_Year + Area +
    Fertilizer + Pesticide +
    Annual_Rainfall,
  data = df
)[,-1]

y <- df$logYield

set.seed(123)

lasso <- cv.glmnet(
  X,
  y,
  alpha = 1,
  nfolds = 10
)
lasso$lambda.min
lasso$lambda.1se
coef(lasso, s = "lambda.1se")
lasso_coef <- coef(
  lasso,
  s = "lambda.1se"
)

lasso_df <- data.frame(
  Variable = rownames(lasso_coef),
  Coefficient = as.numeric(lasso_coef)
)

lasso_df <- subset(
  lasso_df,
  Coefficient != 0
)

lasso_df
nrow(lasso_df)-1
lasso_df <- lasso_df[
  order(abs(lasso_df$Coefficient),
        decreasing = TRUE),
]

lasso_df
pred_lasso <- predict(
  lasso,
  newx = X,
  s = "lambda.1se"
)

library(Metrics)

rmse(  y, pred_lasso)
SSE <- sum((y - pred_lasso)^2)
SST <- sum((y - mean(y))^2)
R2 <- 1 - SSE/SST
R2

plot(lasso$glmnet.fit,
     xvar = "lambda",
     label = TRUE)
plot(lasso)

df %>%
  group_by(Crop) %>%
  summarise(
    MeanYield = mean(Yield),
    MedianYield = median(Yield)
  ) %>%
  arrange(desc(MeanYield))


#table
pred_ridge <- as.vector(pred_ridge)
pred_lasso <- predict(
  lasso,
  newx = X_test,
  s = "lambda.min"
)
pred_lasso <- as.numeric(pred_lasso)
RMSE(pred_lasso, y_test)
R2(pred_lasso, y_test)

library(Metrics)
library(caret)

model_comparison <- data.frame(
  Model = c(
    "Linear Regression",
    "Random Forest",
    "Ridge Regression",
    "LASSO Regression"
  ),
  RMSE = c(
    RMSE(pred_lm, test$logYield),
    RMSE(pred_rf,test$Yield),
    RMSE(pred_ridge, y_test),
    RMSE(pred_lasso, y_test)
  ),
  R_Squared = c(
    R2(pred_lm, test$logYield),
    R2(pred_rf,test$Yield),
    R2(pred_ridge, y_test),
    R2(pred_lasso, y_test)
  )
)

model_comparison <- model_comparison[
  order(model_comparison$RMSE),
]

model_comparison
library(knitr)

kable(
  model_comparison,
  digits = 4,
  caption = "Comparison of Prediction Models"
)

