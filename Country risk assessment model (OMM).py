#####################################
### AN ORDERED MULTINOMINAL MODEL ###
#####################################

import pandas as pd
import numpy as np
import datetime as dt
import sklearn
import warnings
warnings.simplefilter(action="ignore", category=Warning)

from datetime import date, datetime
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import KNNImputer   # KNN algorithm was used to impute missing variabels.

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.width', 500)

data1 = "Model_1_Prediction_Set"  # here will be the name of the dataset of 200 countries to be estimated (must be in xlsx format)
data2 = "Model_2_Prediction_Set"  # here will be the name of the dataset of ~40 countries to be estimated (must be in xlsx format)
model1 = "Model_1_Train_Set"  # Train-Test dataset for 200 countries (must be in .xlsx format)
model2 = "Model_2_Train_Set"  # Train-Test dataset for ~40 countries (must be in .xlsx format)
dependent = "Dependent_var"  # Dependent variable (must be in .xlsx format)
merger = "Ulke Listesi Exim"    # Key table (must be in .xlsx format)


def OMM_logit(data1, data2):

    # Model 1

    # Read the Excel sheet named Monthly and drop the first row and five columns, use the thrid row as a header, also drop the blank row between header and data
    df = pd.read_excel(f"{model1}.xlsx", "Monthly",
                       usecols=lambda x: x not in ["Data Item Definition", "Notes", "First Forecast Year", "Last Reviewed", "Source"],
                       header=3).drop(0, 0)
    # Exchanging the direction columns and rows.
    df = df.melt(id_vars=['Geography', 'Data Item']).pivot(index=['Geography', 'variable'], columns='Data Item').reset_index()

    # Initially we have columns like this January 2022, February 2022. We convert them as 2022-01-30, 2022-02-30 accordingly. Then set as index both along with Geography (Country) variable.
    df['variable'] = pd.to_datetime(df['variable']).dt.to_period('M').dt.to_timestamp('M')
    df = df.set_index(['Geography', 'variable']).sort_index().reset_index()

    # Due to the above conversion, we now have two level column index. The below code checks if the first level is emptly. If not, the first row is used as column name, otherwise the second level is used.
    df.columns = df.columns.map(lambda x: x[1] if x[1] else x[0])
    df = df.rename(columns={"Geography": "COUNTRIES"})

    # Previously determined variables based on actuary principles.
    secilmis_var = ['variable','COUNTRIES','Crime and Security Risk Index', 'Labour Market Risk Index', 'Logistics Risk Index', 'LTER Economic Growth', 'LTER External Factors', 'LTER Financial Markets', 'LTER Fiscal Policy', 'LTER Monetary Policy', 'LTER Structural Characteristics', 'LTPR Characteristics of Polity', 'LTPR Characteristics of Society', 'STPR Policy Continuity', 'STPR Security/External Threats', 'Operational Risk Index', 'STPR Social Stability', 'STPR Policy-Making Process', 'STER Financial Markets', 'STER Fiscal Policy', 'STER Monetary Policy']
    df.columns= df.columns.str.replace(",","")
    df = df.loc[:, df.columns.isin(secilmis_var)]

    # Below data contains common variables of the datasets we use in this prediction.
    ulke_kod = pd.read_excel(f"{merger}.xlsx")

    # Recently Turkey has changed its name as Turkiye. For now, I continue use Turkey and change it at final stage.
    df["COUNTRIES"] = df["COUNTRIES"].str.replace('Turkiye', 'Turkey')

    # If there is a mismatch between datasets or an error occurs, the below two lines give you notice without interrupting the process.
    drop_list = df[~df["COUNTRIES"].isin(ulke_kod["COUNTRIES"].unique().tolist())]["COUNTRIES"].unique().tolist()
    print(f"Those who are excluded from the dataset because the country code does not match. Need attention!:\n {drop_list}")

    df = pd.merge(df, ulke_kod, on="COUNTRIES", how="left").dropna()

    # Read the dependent variable.
    Dep_var = pd.read_excel(f"{dependent}.xlsx")

    # Select columns of interest
    Dep_var = Dep_var[["ULKE KODU EXIM", "ULKE", "Risk Dönemi", "Risk Class", "Release_Date"]]

    # I have only the release date of the dependent variables. I also require an end date to define a time interval that encompasses the independent variables.
    Dep_var.loc[Dep_var.iloc[:, 2] != Dep_var.iloc[:, 2].shift(1), "Rel_Date"] = Dep_var.loc[Dep_var.iloc[:, 2] != Dep_var.iloc[:, 2].shift(1), "Release_Date"].shift(-1)
    for i in Dep_var.loc[:, Dep_var.columns.str.contains("Date")].columns:
        Dep_var[f'{i}'] = Dep_var[f'{i}'].astype("datetime64")
    Dep_var.loc[Dep_var["Release_Date"] == Dep_var["Release_Date"].max(), "Rel_Date"] = pd.to_datetime("today").strftime("%Y-%m-%d")
    Dep_var["Rel_Date"] = Dep_var["Rel_Date"].fillna(method='ffill')

    # Let's merge dependent and independent based on the time interval I created above, and do some data editing.
    df = df[df["variable"] >= "01/24/2017"]
    Dep_var.rename(columns={"ULKE KODU EXIM": "ÜLKE KODU"}, inplace=True)
    df = df.merge(Dep_var, on="ÜLKE KODU", how="left").query('(Rel_Date> variable) & (Release_Date <= variable)')
    df.drop_duplicates(inplace=True)
    df.set_index("ÜLKE KODU", inplace=True)

    df.drop(["variable", "COUNTRIES","Risk Dönemi", "Release_Date", "Rel_Date", "ÜLKELER", "ULKE"], 1, inplace=True)

    df.replace("-", np.nan, inplace=True)
    df.dropna(inplace=True)

    # To be sure that except the dependent variable, all variables' format is float. (Requirment of OMM)
    for i in df.iloc[:, df.columns != "Risk Class"].columns:
        df[f'{i}'] = df[f'{i}'].astype("float")

    # To be sure the dependent variable is assigned as categorical. (Requirment of OMM)
    df["Risk Class"] = df["Risk Class"].astype("Int64").astype("category")

    # Our data for Model 1 is ready. Let's train Model 1.

    y = df["Risk Class"]
    X = df.loc[:, df.columns != "Risk Class"]

    # Split the dataset as train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Fit model and get parameters
    mod_logit = OrderedModel(y_train,
                              X_train,
                              distr='logit', disp=False)

    res_logit_train = mod_logit.fit(method='bfgs')

    pred_y_train = res_logit_train.model.predict(res_logit_train.params, exog=X_train)
    pred_choice_train = pred_y_train.argmax(1) #y_pred_train
    print('Fraction of correct choice predictions for train data set (Model 1)')
    print((np.asarray(y_train.values.codes) == pred_choice_train).mean())

    pred_y_test = res_logit_train.model.predict(res_logit_train.params, exog=X_test)
    pred_choice_test = pred_y_test.argmax(1)  # y_pred_test
    print('Fraction of correct choice predictions for test data set (Model 1)')
    print((np.asarray(y_test.values.codes) == pred_choice_test).mean())

    # Prepare data1 for prediction (It is pretty much the same what we have done with Model1 above. Just the data itself different.

    # Read the Excel sheet named Monthly and drop the first row and five columns, use the thrid row as a header, also drop the blank row between header and data
    df1 = pd.read_excel(f"{data1}.xlsx", "Monthly",
                       usecols=lambda x: x not in ["Data Item Definition", "Notes", "First Forecast Year", "Last Reviewed", "Source"],
                       header=3).drop(0, 0)

    # Exchanging the direction columns and rows.
    df1 = df1.melt(id_vars=['Geography', 'Data Item']).\
        pivot(index=['Geography', 'variable'], columns='Data Item').reset_index()

    # Due to the above conversion, we now have two level column index. The below code checks if the first level is emptly. If not, the first row is used as column name, otherwise the second level is used.
    df1.columns = df1.columns.map(lambda x: x[1] if x[1] else x[0])
    df1 = df1.rename(columns={"Geography": "COUNTRIES"})

    # Making some editing and prepearing the data for prediction.
    df1 = df1.merge(ulke_kod.iloc[:, [0,2]], on="COUNTRIES", how="left")
    df1.drop(["COUNTRIES","variable"], 1, inplace=True)
    df1.set_index("ÜLKE KODU", inplace=True)
    df1 = df1.replace("-", np.nan).dropna()

    # To be sure all variables' format are float. (Requirment of OMM)
    for i in df1.columns:
        df1[f'{i}'] = df1[f'{i}'].astype("float")

    df1.reset_index(inplace=True)
    dfa1 = df1.groupby("ÜLKE KODU").mean()
    dfa1.columns= dfa1.columns.str.replace(",","")
    dfa1 = dfa1.loc[:, dfa1.columns.isin(secilmis_var)]

    # Make predicition based on the above fitted OMM.
    result = res_logit_train.predict(dfa1)
    result.columns = ["1", "2", "3", "4", "5", "6", "7", "8"]
    pred_y = res_logit_train.model.predict(res_logit_train.params, exog=dfa1)
    result["Max_Prob_Class"] = pred_y.argmax(1)+1
    result["Source"] = "Model1"

    # Model 2

    # Below code reads the training data of Model2, and merge with the dependent variable
    cred = pd.read_excel(f'{model2}.xlsx', dtype={"Risk Class": "Int64"}). \
        rename(columns={"ULKE KODU EXIM": "ÜLKE KODU"}). \
        merge(Dep_var, on="ÜLKE KODU", how="left"). \
        query('(Rel_Date> update) & (Release_Date <= update)'). \
        set_index("ÜLKE KODU"). \
        astype({"Risk Class": "Int64"})

    cred= cred.replace("-", np.nan).select_dtypes(include=np.number)

    # There are some missing observation. We need to imput missing data by KNN algorithm.
    imputer = KNNImputer(n_neighbors=2)
    # impute missing values
    cred = pd.DataFrame(imputer.fit_transform(cred), columns=cred.columns).set_index(cred.index)

    # To be sure that except the dependent variable, all variables' format is float. (Requirment of OMM)
    for i in cred.iloc[:, cred.columns != "Risk Class"].columns:
        cred[f'{i}'] = cred[f'{i}'].astype("float")

    # To be sure your dependent variable is categorical
    cred["Risk Class"] = cred["Risk Class"].astype("Int64").astype("category")


    # Let's train Model 2.
    y = cred["Risk Class"]
    X = cred.loc[:, cred.columns != "Risk Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    mod_logit = OrderedModel(y_train,
                             X_train,
                             distr='logit', disp=False)
    res_logit_train = mod_logit.fit(method='cg')

    pred_y_train = res_logit_train.model.predict(res_logit_train.params, exog=X_train)
    pred_choice_train = pred_y_train.argmax(1)  # y_pred_train
    print('Fraction of correct choice predictions for train data set (Model2)')
    print((np.asarray(y_train.values.codes) == pred_choice_train).mean())

    pred_y_test = res_logit_train.model.predict(res_logit_train.params, exog=X_test)
    pred_choice_test = pred_y_test.argmax(1)  # y_pred_test
    print('Fraction of correct choice predictions for test data set (Model2)')
    print((np.asarray(y_test.values.codes) == pred_choice_test).mean())

    # Let's predict Data2 by Model 2 which has just been fitted above.
    da2 = pd.read_excel(f"{data2}.xlsx", index_col="ULKE KODU EXIM"). \
        rename_axis("ÜLKE KODU"). \
        drop(['COUNTRIES', 'update', 'ULKE KODU'], axis=1). \
        replace("-", np.nan). \
        astype(float)

    # There are some missing observation. Since Data2 is very short, we need to imput missing data by KNN algorithm.
    imputer = KNNImputer(n_neighbors=2)
    # impute missing values
    da2 = pd.DataFrame(imputer.fit_transform(da2), columns=da2.columns).set_index(da2.index)

    # Take the countries code that was predicted by Model 1, and exclude them from the prediction dataset of Model 2
    ulke_kodu_list = result.index.unique().tolist()
    da2 = da2[(~da2.index.isin(ulke_kodu_list)) & (~da2.index.isnull())]

    # Make predicition based on the above fitted OMM of Model 2
    result2 = res_logit_train.predict(da2)
    result2.columns = ["1", "2", "3", "4", "5", "6", "7", "8"]
    pred_y = res_logit_train.model.predict(res_logit_train.params, exog=da2)
    result2["Max_Prob_Class"] = pred_y.argmax(1)+1
    result2["Source"] = "Model2"

    # Concat the result as a unique dataframe
    result_son = pd.concat([result, result2])
    result_son = pd.merge(result_son.reset_index(),
                          ulke_kod.iloc[:, [0,1]], right_on="ÜLKE KODU", left_on="ÜLKE KODU", how="left")
    result_son.set_index(["ÜLKE KODU", "ÜLKELER"], 1, inplace=True)

    # Add the timestamp
    now = datetime.now()
    now = now.strftime("%d_%m_%Y %H_%M_%S")

    # Count how many countries were predicted by what model.
    Model1_ulke = result.reset_index()["ÜLKE KODU"].nunique()
    Model2_ulke = result2.reset_index()["ÜLKE KODU"].nunique()

    # Detect what countries could not be predicted.
    basarısız= ulke_kod[~ulke_kod["ÜLKE KODU"].isin(result_son.reset_index()["ÜLKE KODU"].to_list())]["ÜLKELER"].unique()

    # Inform the user about the successful and unsuccesful result.
    print(f"\nNumber of countries predicted with Model 1 is {Model1_ulke},\n"
          f"Number of countries predicted with Model 2 is {Model2_ulke},\n"
          f"Total number of predicted countries is {Model1_ulke+Model2_ulke}.\n"
          f"Countries with failed predictions: {basarısız} ")

    # Inform the user that process is complated.
    print(f"\nThe output is saved in the folder with the name Logit_results_{now}.xlsx.\n")

    # Save the results as Excel file.
    result_son.to_excel(f'Logit_results_{now}.xlsx')


##### OMM FUNCTION ####
OMM_logit(data1, data2)
#######################


