# Country-risk-assessment-model

Please imagine you work in an insurance company whose mission is to ensure the country’s exporters' risk abroad, and you want to develop an insurance pricing strategy. One of the critical components of your pricing strategy should be the evaluation of the risk associated with the buyer firms' country. For this purpose, I build a country risk assessment model using ordered logistic regression.

The following table is presented as an example to illustrate the expected outcome. In our case, there are 8 classes (1 is the best, 8 is the worst). Each class column (1-8) shows the probability that the relevant country belongs to that class. For example, according to Model 1, Denmark belongs to the Class 2 with a probability of 70%, while it belongs to the Class 1 with a probability of 21%.

![image](https://user-images.githubusercontent.com/94282435/221692168-7bab75f9-31d4-4f2d-a59f-357236845b12.png)

There are two models in the attached script (please see the Source column above). Both works based on the same principles but with different datasets. My aim is to make production for approximately 240 countries. However, one dataset has information on 200 countries and the other has information on approximately 40 countries. So, I run two models. At the end of the process, I will have a combined outcome for all ~240 countries like the one above.
