setwd("C://Users//SHUBHANGI//Downloads")

df <- read.csv("SUNPHARMA.NS.csv")
df$date <- as.Date(df$Date, "%m-%d-%Y")

list_var <- c("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume")

train_data <- df[c(1:3550),]
test_data <- df[c(3551:5071),]

train_data <- train_data[,which(names(train_data) %in% list_var)]
test_data <- test_data[,which(names(test_data) %in% list_var)]

model <- lm(Close ~ Open + Volume , data = train_data )
summary(model)
AIC(model)
library(usdm)
vif(model)

qqnorm(model$residuals)
predict<- predict(model, test_data)
predicted <- as.data.frame(predict(model, test_data))
test_data <- df[c(3551:5071),]

library(MLmetrics)
MAPE(y_pred = predict, y_true = test_data$Close)
RMSE(y_pred = predict, y_true = test_data$Close)

library(e1071)
tuneResult <- tune(svm, Close ~ Open + Shares.Traded ,  data = train_data,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
)
print(tuneResult)


# Draw the tuning graph
plot(tuneResult)
tunedModel <- tuneResult$best.model
tunedModelY <- predict(tunedModel, test_data) 

error <- test_data$Close - tunedModelY
RMSE(y_pred = tunedModelY, y_true = test_data$Close)

X <- cbind(test_data$date,predicted)
Y <- cbind(X,tunedModelY)
write.csv(X ,file = "SUNPHARMA graph.csv",row.names = F)






