# **Machine Learning Lab1**

Logistic Regression

By He Weiyi PB20051035

## **1 The loss curve of one train process**

![picture 3](../images/b8b180efc8d381030a5a81b00c0395ccffe54e993794488257b68304535b8f26.png)  

## **2 The comparation table of different parameters**

l1 and l2 show the method of the regularzation, the number in the blank shows the parameter lamada of the regularzation. After doing 5 times bootstrapping and get the mean, we have the comparation table as below:


 Train Accuracy | Test Accuracy | l1 | l2 | learning rate | iteration
----------------|---------------|----|----|---------------|----------
 0.8004 | 0.8131 | / | / | 0.01 | 10000
 0.6938 | 0.7104 | / | / | 0.01 | 1000
 0.8021 | 0.8086 | / | / | 0.01 | 100000
 0.8174 | 0.7941 | 1 | / | 0.01 | 10000
 0.7971 | 0.8295 | / | 1 | 0.01 | 10000
 0.7104 | 0.6904 | / | / | 0.001 | 10000
 0.8017 | 0.8053 | / | 2 | 0.01 | 10000
 0.8065 | 0.8104 | / | / | 0.05 | 10000
 0.7942 | 0.7816 | / | 1 | 0.01 | 5000
 0.8050 | 0.8154 | / | 1 | 0.01 | 20000
 
## **3 The best accuracy of test data**

* We use accuracy as the metric, and among all different parameters we train with, the best accuracy of test data is 0.8295, (By using 5 times bootstrapping) with parameters: l2 regularization(lamada=1), learning rate 0.01 and 10000 iteration times.