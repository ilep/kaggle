

library(data.table)
library(ggplot2)
library(Hmisc)


SALES_PREDICTION_PATH = "C:/Users/user/Documents/workspace/ilepStats/kaggle/sales_prediction/"


stores = read.csv("C:/Users/user/Documents/workspace/ilepStats/kaggle/sales_prediction/sales-prediction-challenge/store.csv")
stores = data.table(stores)
head(stores)

train = read.csv("C:/Users/user/Documents/workspace/ilepStats/kaggle/sales_prediction/sales-prediction-challenge/train.csv")
train = data.table(train)



dim(train)
# [1] 1017209       9


unique(train$Store)
# 1115 unique stores


train[,Date:= as.Date(Date)]

min(train$Date) # "2013-01-01"
max(train$Date) # "2015-07-31"


store_ex = 1

sales_trend = train[Store == store_ex,list(Date, Sales)]
sales_trend[,date_num:=as.numeric(Date)]
sales_trend = sales_trend[order(date_num),list(Date,Sales, cumsum_sales=cumsum(Sales))]
plot(sales_trend[,list(Date,cumsum_sales )], type = 'l', lwd = 1)



all_cumsums = train[,list(date_num = as.numeric(Date), Date, Sales),by=c('Store')][order(date_num),][,list(date_num, Date, Sales, Sales.cumsum = cumsum(Sales)),by=c('Store')]
stopifnot(identical(sort(unique(all_cumsums$Store)), unique(train$Store)))

all_cumsums = all_cumsums[Store %in% c(1,2),]

g <- ggplot(data=all_cumsums,aes(x=Date, y=Sales.cumsum, group = Store)) + geom_line(alpha=0.1, colour = 'red') + xlab("Date") + ylab("cumul des ventes")
g <- g + ggtitle("Sales cumsums for each store") 

# g <- g + stat_summary(fun.data ="mean_sdl", mult=1, geom = "smooth")
# g <- g + stat_summary(fun.y=mean, geom="line", colour="green") 
g + ggsave(sprintf("%ssales_cumsums.jpeg",SALES_PREDICTION_PATH),width=100,height=100)

ggplot(all_cumsums,aes(x=Date,y=Sales.cumsum)) + stat_summary(fun.data ="mean_sdl", mult=1, geom = "smooth")


# distribution des ventes par jour de la semaine
boxpl <- ggplot(train, aes(y =Sales, x = as.character(DayOfWeek), fill = factor(DayOfWeek))) + geom_boxplot()


train

plot(train$Customers, train$Sales)


ggplot(data = train, aes(x = Customers,y = Sales, group = Store, colour = factor(Store))) + geom_point(alpha=0.1)


# relation between Customers and Sales; bigger traffic ==> bigger Sales
plot(train[Store==1,Customers], train[Store==1,Sales], pch = 19, col = "red")






