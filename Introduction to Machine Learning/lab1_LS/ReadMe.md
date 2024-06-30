# **Machine Learning Lab1**

Logistic Regression

By He Weiyi PB20051035

## **1 Introduction to Logistic Regression**
We use the dataset to train a model that can decide whether to lend money to a person or not. With the sigmoid function, we build a connection between the input (all the attributes of a person) and the classification probability. Then by maximizing the likelihood function we get the optimal parameters as our model parameters. Finally by running the model on the test data we evaluate our model.

## **2 My Work**
I choose to use the frame supplied.

1. In Logistic.py, there is my own Logistic Regression class. Pay attention that the name and the parameters of the function in the class is possibly a little different with the frame supplied, since I think it could make the code more clear.

2. In Loan.ipynb, first I deal with NULL rows by dropping them so that the number of the sample turn to 480. Then I encode all the attributes (which is not numbers) by using label encoding. I use the Bootstrapping method to split the dataset, and also I use "Min-Max" method to do the normalization.

    Then we do the traning.

    To reach the basic requirement, I use the loss function without regularization (In the following part we will compare it with l1 and l2 regularization) and plot the loss curve of training (as one training process shown in the report). We use gradient descent method to get the optimal solution and set a tol to stop the iteratation.

    When I was evaluating my model, I found that the accuracy may be more convincing if I get more than one train set and use the mean accuracy. So I use the bootstrapping 5 times to do my evaluation, this work is shown in the last part of Loan.ipynb. In fact, we can change the bootstrapping times to any other numbers.

## **3.The training with different parameters**
In Loan.ipynb following the instructions one can easily change the parameters to see the accuracy, there are only some examples in the Loan.ipynb and
1. l1 regularization
2. l2 regularization
3. learning rate
4. iteration times

   In Loan.ipynb following the instructions one can easily change the parameters to see the accuracy. There are only a small part of examples in the Loan.ipynb and all the comparison of the parameters are in a table in the Report.pdf.