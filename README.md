# Country-risk-assessment-model

Please imagine you work in an insurance company whose mission is to ensure the country’s exporters' risk abroad, and you want to develop an insurance pricing strategy. One of the critical components of your pricing strategy should be the evaluation of the risk associated with the buyer firms' country. For this purpose, I build a country risk assessment model using ordered logistic regression.

The following table is presented as an example to illustrate the expected outcome. In our case, there are 8 classes (1 is the best, 8 is the worst). Each class column (1-8) shows the probability that the relevant country belongs to that class. For example, according to Model 1, Denmark belongs to the Class 2 with a probability of 70%, while it belongs to the Class 1 with a probability of 21%.)

![Screen Shot 2023-04-24 at 11 07 47](https://user-images.githubusercontent.com/94282435/234038296-5d17eabe-3de5-4e5e-a07b-ca23869702af.png)

There are two models in the attached script (please see the Source column above). Both works based on the same principles but with different datasets. My aim is to make prediction for approximately 240 countries. However, one dataset has information on 200 countries and the other has information on approximately 40 countries. So, I run two models. At the end of the process, I will have a combined outcome for all ~240 countries like the one above.

As it is seen in the attached Python script, my code starts with the below descriptions:

![Screen Shot 2023-04-22 at 19 57 13](https://user-images.githubusercontent.com/94282435/233812352-8c90f3bd-ddde-444a-9840-209f2925c68e.png)

It makes the script user-friendly because when an update is required, even a non-code literate person can run the model by changing the definitions above. For example, assuming the working directory remains the same, if my junior colleague wants to run the models for new datasets, it is just needed to replace the descriptions of data1 and data2 with the new excel files’ name. Let’s say we have two new excel files to run regression whose names are “New_data_1.xlsx” and “New_data_2.xlsx”. So, my junior colleague just needs to rewrite the definition of data1 and data2 as "New_data_1" and "New_data_2" accordingly, and then run a single line of code and have the outcome.

In brief, the function I created and named as "OMM_logit" [OMM_logit(data1, data2)] takes in two sets of data as input because there are two models combined. Assuming the above mentioned datasets are already located in the current working directory and the file paths in the attached script have been properly updated, anyone can run the regression typing a single line of code: OMM_logit(data1, data2). The function will train an ordered multinominal logistic model using the "model1" and "model2" datasets and make prediction using the “data1” and “data2” datasets, and the resulting output will be saved in the current working directory with a file name indicating the time it was generated. The title below says it was generated on January 24, 2023 at 11:12:25 AM.

                         Logit_results_24_01_2023 11_12_25.xlsx


The execution of the code

What I see on my console during the projection process is as follows. 

![Screen Shot 2023-04-24 at 11 09 32](https://user-images.githubusercontent.com/94282435/234038782-ef51aae8-b874-40ab-9935-e500a02d9642.png)![Screen Shot 2023-04-24 at 11 09 43](https://user-images.githubusercontent.com/94282435/234038801-be83398f-2dbe-4d37-a075-6ab0da114647.png)



