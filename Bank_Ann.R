#importing the data from system drive

bank <-read.csv("D:/RTUTORIAL/R Project/bank.csv",na.strings = c(" ","","NA"))
bank_train<-bank

#checking column wise missing values
colSums(is.na(bank))

#making pdays variable is NULL because we dont use that variable
bank_train$pdays<-NULL

#imputing missing variables by using KNN imputation

library(VIM)
bank_train<-kNN(bank_train,c('housing'),k=sqrt(10))
bank_train<-kNN(bank_train,c('job'),k=sqrt(10))
bank_train<-kNN(bank_train,c('marital'),k=sqrt(10))
bank_train<-kNN(bank_train,c('education'),k=sqrt(10))
bank_train<-kNN(bank_train,c('default'),k=sqrt(5))
bank_train<-kNN(bank_train,c('loan'),k=sqrt(10))

#cross checking the missing values after performing  KNN imputation
colSums(is.na(bank_train))

#After KNN imputation some extra varlables are generated we have to make them NULL

bank_train$housing_imp<-NULL
bank_train$job_imp<-NULL
bank_train$marital_imp<-NULL
bank_train$education_imp<-NULL
bank_train$default_imp<-NULL
bank_train$loan_imp<-NULL
# Encoding job variable from categorical to factors

table(bank_train$job)
bank_train$job = factor(bank_train$job,
                  levels = c('admin.', 'blue-collar', 'entrepreneur',
                             'housemaid','management','retired',
                             'self-employed','services','student',
                             'technician','unemployed'),
                  labels = c(1,2,3,4,5,6,7,8,9,10,11))

#Encoding marital variable

table(bank_train$marital)
bank_train$marital = factor(bank_train$marital,
                      levels = c('divorced','married','single'),
                      labels = c(1, 2, 3))

#Encoding education variable

table(bank_train$education)
bank_train$education = factor(bank_train$education,
                        levels = c('basic.4y','basic.6y','basic.9y',
                                   'high.school','illiterate','professional.course',
                                   'university.degree'),
                        labels = c(1,2,3,4,5,6,7))

#Encoding default varible

table(bank_train$default)
bank_train$default = factor(bank_train$default,
                      levels = c('no','yes'),
                      labels = c(0,1))

#Encoding housing variable

table(bank_train$housing)
bank_train$housing = factor(bank_train$housing,
                      levels = c('no','yes'),
                      labels = c(0,1))

#Encoding loan variable

table(bank_train$loan)
bank_train$loan = factor(bank_train$loan,
                   levels = c('no','yes'),
                   labels = c(0,1))

#Encoding contact variable

table(bank_train$contact)
bank_train$contact = factor(bank_train$contact,
                      levels = c('cellular','telephone'),
                      labels = c(0,1))

#Encoding month variable

table(bank_train$month)
bank_train$month = factor(bank_train$month,
                    levels = c('mar','apr','may','jun','jul',
                               'aug','sep','oct','nov','dec'),
                    labels = c(3,4,5,6,7,8,9,10,11,12))

#Encoding day_of_week variable

table(bank_train$day_of_week)
bank_train$day_of_week = factor(bank_train$day_of_week,
                          levels = c('mon','tue','wed','thu','fri'),
                          labels = c(1,2,3,4,5))

#Encoding poutcome variable

table(bank$poutcome)
bank_train$poutcome = factor(bank_train$poutcome,
                       levels = c('failure','nonexistent','success'),
                       labels = c(1,0,2))

#Encoding Y variable

table(bank_train$y)
bank_train$y = factor(bank_train$y,
                levels = c('no','yes'),
                labels = c(0,1))

#we have to make all factor variables to nueric by using as.numeric

bank_train$job<-as.numeric(bank_train$job)
bank_train$marital<-as.numeric(bank_train$marital)
bank_train$education<-as.numeric(bank_train$education)
bank_train$default<-as.numeric(bank_train$default)
bank_train$housing<-as.numeric(bank_train$housing)
bank_train$loan<-as.numeric(bank_train$loan)
bank_train$contact<-as.numeric(bank_train$contact)
bank_train$month<-as.numeric(bank_train$month)
bank_train$day_of_week<-as.numeric(bank_train$day_of_week)
bank_train$poutcome<-as.numeric(bank_train$poutcome)

#Splitting the data into training and testing set by using caTools

library(caTools)
set.seed(123)
split = sample.split(bank_train$y, SplitRatio = 0.75)
trainn_set = subset(bank_train, split == TRUE)
testn_set = subset(bank_train, split == FALSE)

# Feature Scaling
trainn_set[-20] = scale(trainn_set[-20])
testn_set[-20] = scale(testn_set[-20])

# Fitting ANN to the Training set by using h2o package
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'y',
                            training_frame = as.h2o(trainn_set),
                            activation = 'Rectifier',
                            hidden = c(200,200),
                            epochs = 200,
                            train_samples_per_iteration = -2,
                            seed = 1122)
plot(model)
h2o.performance(model)

#checking the variable importants 

h2o.varimp_plot(model,num_of_features = 20)

# Predicting the Test set results

predict.dl2 <- as.data.frame(h2o.predict(model, as.h2o(testn_set)))
predict.dl2$predict
h2o.shutdown()

# Making the Confusion Matrix

cmm = table(testn_set$y,predict.dl2$predict)
cmm

#performing logistic regression

classifier = glm(formula = y ~ .,
              family = binomial,
              data = trainn_set)

# Predicting the Test set results

pred = predict(classifier, type = 'response', newdata = testn_set[-20])
pred = ifelse(pred > 0.5, 1, 0)

# Making the Confusion Matrix

cm = table(testn_set[,20], pred)
cm

