#libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

#global variables
DelDict = {"IN": 1, "TR": 1, "HY": 2, "BL": 2, "WB": 3, "ON": 3}
GenDict = {"F": 1, "M": 2}
GradeDict = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1, "W": 0}
TermDict = {"FA": 0.7, "SP": 0, "SU": 0.3}
EHIDict = {"Low": 1, "Medium-Low": 2, "Medium": 3, "Medium-High": 4, "High": 5}
group_names = ["<18", "18-22", "23-26", "27-30", "31-34", "35-38", ">38"]
bins = [0, 18, 23, 27, 31, 35, 38, 100]

#main script
def main():
   #import data files
   
   print("So you want to ruin your computer...")
   print("Let's start by reading in the CSV files")
   dfCsans = pd.read_csv("./CHsans.csv", dtype = "string")
   dfCwith = pd.read_csv("./CHwith.csv")
   dfPsans = pd.read_csv("./PYsans.csv")
   dfPwith = pd.read_csv("./PYwith.csv")
   dfEsans = pd.read_csv("./ENGsans.csv")
   dfEwith = pd.read_csv("./ENGwith.csv")
   
   print("Now let's clean the files")
      #clean datafiles
        #drop NA values
   dfCsans.dropna(inplace = True)
   dfCwith.dropna(inplace = True)
   dfPsans.dropna(inplace = True)
   dfPwith.dropna(inplace = True)
   dfEsans.dropna(inplace = True)
   dfEwith.dropna(inplace = True)
   
        #drops STC SEC NAME and Section columns
   dfCwith.drop(columns = ["Stc Sec Name", "Section"], inplace = True)
   dfCsans.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
   dfPwith.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
   dfPsans.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
   dfEwith.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
   dfEsans.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
      
     #Clean up grades
   dfCsans["Grade"] = dfCsans["Grade"].str[0]
   dfCwith["Grade"] = dfCwith["Grade"].str[0]
   dfPsans["Grade"] = dfPsans["Grade"].str[0]
   dfPwith["Grade"] = dfPwith["Grade"].str[0]
   dfEsans["Grade"] = dfEsans["Grade"].str[0]
   dfEwith["Grade"] = dfEwith["Grade"].str[0]
   
     #drops D, F, W students in MAT-171
   dfCwith = dummy_drop(dfCwith)
   dfPwith = dummy_drop(dfPwith)
   dfEwith = dummy_drop(dfEwith)

      #grades to numbers
   dfCsans = grades_to_numbers(dfCsans)   
   dfCwith = grades_to_numbers(dfCwith)
   dfPsans = grades_to_numbers(dfPsans)
   dfPwith = grades_to_numbers(dfPwith)
   dfEsans = grades_to_numbers(dfEsans)
   dfEwith = grades_to_numbers(dfEwith)
   
        #converts delivery method to numbers
   dfCsans = devilery_to_numbers(dfCsans)
   dfCwith = devilery_to_numbers(dfCwith)
   dfPsans = devilery_to_numbers(dfPsans)
   dfPwith = devilery_to_numbers(dfPwith)
   dfEsans = devilery_to_numbers(dfEsans)
   dfEwith = devilery_to_numbers(dfEwith)
   
        #convers gender to numbers
   dfCsans = gender_to_numbers(dfCsans)
   dfCwith = gender_to_numbers(dfCwith)
   dfPsans = gender_to_numbers(dfPsans)
   dfPwith = gender_to_numbers(dfPwith)
   dfEsans = gender_to_numbers(dfEsans)
   dfEwith = gender_to_numbers(dfEwith)

        #convers EHI to numbers
   dfCsans = EHI_to_numbers(dfCsans)
   dfCwith = EHI_to_numbers(dfCwith)
   dfPsans = EHI_to_numbers(dfPsans)
   dfPwith = EHI_to_numbers(dfPwith)
   dfEsans = EHI_to_numbers(dfEsans)
   dfEwith = EHI_to_numbers(dfEwith)
   
        #creates year and term columns
   dfCsans["Year"] = dfCsans["Stc Term"].str[0:4].astype(int)
   dfCsans["Term"] = dfCsans["Stc Term"].str[4:6]
   dfCwith["Year"] = dfCwith["Stc Term"].str[0:4].astype(int)
   dfCwith["Term"] = dfCwith["Stc Term"].str[4:6]
   dfPsans["Year"] = dfPsans["Stc Term"].str[0:4].astype(int)
   dfPsans["Term"] = dfPsans["Stc Term"].str[4:6]
   dfPwith["Year"] = dfPwith["Stc Term"].str[0:4].astype(int)
   dfPwith["Term"] = dfPwith["Stc Term"].str[4:6]
   dfEsans["Year"] = dfEsans["Stc Term"].str[0:4].astype(int)
   dfEsans["Term"] = dfEsans["Stc Term"].str[4:6]
   dfEwith["Year"] = dfEwith["Stc Term"].str[0:4].astype(int)
   dfEwith["Term"] = dfEwith["Stc Term"].str[4:6]
   
        #creates numberic year-term for analysis
   dfCsans = year_term(dfCsans)
   dfCwith = year_term(dfCwith)
   dfPsans = year_term(dfPsans)
   dfPwith = year_term(dfPwith)
   dfEsans = year_term(dfEsans)
   dfEwith = year_term(dfEwith)
   
        #convert objects to integers
   dfCsans["Year"] = dfCsans["Year"].astype(str).astype(int)
   dfCwith["Year"] = dfCwith["Year"].astype(str).astype(int)
   dfPsans["Year"] = dfPsans["Year"].astype(str).astype(int)
   dfPwith["Year"] = dfPwith["Year"].astype(str).astype(int)
   dfEsans["Year"] = dfEsans["Year"].astype(str).astype(int)
   dfEwith["Year"] = dfEwith["Year"].astype(str).astype(int)
   
        #creates age column
   dfCsans["Age"] = dfCsans["Year"] - dfCsans["Year of Birth"].astype(int)
   dfCwith["Age"] = dfCwith["Year"] - dfCwith["Year of Birth"].astype(int)
   dfPsans["Age"] = dfPsans["Year"] - dfPsans["Year of Birth"].astype(int)
   dfPwith["Age"] = dfPwith["Year"] - dfPwith["Year of Birth"].astype(int)
   dfEsans["Age"] = dfEsans["Year"] - dfEsans["Year of Birth"].astype(int)
   dfEwith["Age"] = dfEwith["Year"] - dfEwith["Year of Birth"].astype(int)
   
        #bin age ranges
   bins = [0, 18, 23, 27, 31, 35, 38, 100]
   dfCsans["AgeBin"] = pd.cut(dfCsans["Age"], bins = bins, labels = group_names)
   dfCwith["AgeBin"] = pd.cut(dfCwith["Age"], bins = bins, labels = group_names)
   dfPsans["AgeBin"] = pd.cut(dfPsans["Age"], bins = bins, labels = group_names)
   dfPwith["AgeBin"] = pd.cut(dfPwith["Age"], bins = bins, labels = group_names)
   dfEsans["AgeBin"] = pd.cut(dfEsans["Age"], bins = bins, labels = group_names)
   dfEwith["AgeBin"] = pd.cut(dfEwith["Age"], bins = bins, labels = group_names)
   
   print("Okay, I just cleaned the data, now we can get to work with plotting")
      #plots grade distributions
   uL = 0.5
   Grade_distr(dfCwith, "CHM-151", "with", uL)
   Grade_distr(dfCsans, "CHM-151", "without", uL)
   Grade_distr(dfPwith, "PHY-151", "with", uL)
   Grade_distr(dfPsans, "PHY-151", "without", uL)
   Grade_distr(dfEwith, "ENG-150", "with", uL)
   Grade_distr(dfEsans, "ENG-150", "without", uL)
   
     #plots grade transfer between courses
   success_generator(dfCwith, "CHM-151")
   success_generator(dfPwith, "PHY-151")
   success_generator(dfEwith, "EGR-150")
        #find the Jaccard score
   
   print("Alright, now the plots are generated, let's do some modeling")

    #####modeling
         #create model sets
   col_list = ["Student ID", "Course Name", "Pass", "Fail", "W", "Year of Birth", "DeliveryN", "GradeN", "GenderN", "Year", "YT", "EHIN", "Age", "AgeBin"]
   Cmodel = dfCwith.loc[:, col_list]
   Pmodel = dfPwith.loc[:, col_list]
   Emodel = dfEwith.loc[:, col_list]

   #pairs up last math grade with last course grade
   Cmodel = coursematch(Cmodel, "CHM-151")
   Pmodel = coursematch(Pmodel, "PHY-151")
   Emodel = coursematch(Emodel, "EGR-150")

   print("Now lets normalize the models")
   
   #set data types
   dict_type = {"Year of Birth": float, "DeliveryN": float, "DelM": float, "GradeN": float, "GenderN": float, "Year": float, "YT": float, "EHIN": float, "Age": float, "dt": float, "GradeM": float}
   Cmodel = Cmodel.convert_dtypes(dict_type)
   Cmodel.reset_index()
   Pmodel = Pmodel.convert_dtypes(dict_type)
   Pmodel.reset_index()
   Emodel = Emodel.convert_dtypes(dict_type)
   Emodel.reset_index()
   
   Cmodel = Cmodel[Cmodel["dt"] > 0]
   Pmodel = Pmodel[Pmodel["dt"] > 0]
   Emodel = Emodel[Emodel["dt"] > 0]

#    #normalize data sets (min-max)
   norm_var = ["Year of Birth", "GradeM", "GradeN", "Year", "YT", "EHIN", "Age", "dt"]
   for column in norm_var:
        Cmodel[column] = (Cmodel[column] - Cmodel[column].min()) / (Cmodel[column].max() - Cmodel[column].min())
   for column in norm_var:
     Pmodel[column] = (Pmodel[column] - Pmodel[column].min()) / (Pmodel[column].max() - Pmodel[column].min())
   for column in norm_var:
        Emodel[column] = (Emodel[column] - Emodel[column].min()) / (Emodel[column].max() - Emodel[column].min())
   
   #save model data
   Cmodel.to_csv("Cmodel.csv")   
   Pmodel.to_csv("Pmodel.csv")   
   Emodel.to_csv("Emodel.csv")
     
   #correlation variables for advanced modeling
   col_var = ["GradeM", "Age", "dt", "GenderN", "DeliveryN", "DelM"]
   
      #printing correlation matrix
   Cmodel.corr(numeric_only = True).to_csv("./ChemistryCorr.csv")
   Corr_df = Cmodel.corr(numeric_only = True)
   corr_plot(Corr_df, "CHM-151")
   Pmodel.corr(numeric_only = True).to_csv("./PhysicsCorr.csv")
   Porr_df = Pmodel.corr(numeric_only = True)
   corr_plot(Porr_df, "PHY-151")
   Emodel.corr(numeric_only = True).to_csv("./EngineeringCorr.csv")
   Eorr_df = Emodel.corr(numeric_only = True)
   corr_plot(Eorr_df, "EGR-150")
   
   print("Setting up training and testing subsets, then running the linear model")  
   
     #linear modeling - Just Math Grade
     
   Cats = ["Pass", "Fail", "W", "GradeN"]
   
     #chemistry   
   ResultsLRS = []   
   for metric in Cats:
        ResultsLRS.append(LR_modeling(Cmodel, metric, "CHM-151", ["GradeM"]))
   lrs_df = pd.DataFrame(ResultsLRS)
   lrs_df.reset_index()
   lrs_df.to_csv("LR_data.csv")
   LR_plots(lrs_df, "R-Squared Values for Modeling only with Math Grade", "CSimpleLinear.png")

     #physics   
   ResultsLRS = []   
   for metric in Cats:
        ResultsLRS.append(LR_modeling(Cmodel, metric, "PHY-151", ["GradeM"]))
   lrs_df = pd.DataFrame(ResultsLRS)
   lrs_df.reset_index()
   lrs_df.to_csv("LR_data.csv")
   LR_plots(lrs_df, "R-Squared Values for Modeling only with Math Grade", "PSimpleLinear.png")
   
     #engineering   
   ResultsLRS = []   
   for metric in Cats:
        ResultsLRS.append(LR_modeling(Cmodel, metric, "EGR-150", ["GradeM"]))
   lrs_df = pd.DataFrame(ResultsLRS)
   lrs_df.reset_index()
   lrs_df.to_csv("LR_data.csv")
   LR_plots(lrs_df, "R-Squared Values for Modeling only with Math Grade", "ESimpleLinear.png")
   
     #robust modeling
   Cats = ["Pass", "Fail", "W", "GradeN"]
   
     #chemistry
   ResultsLR = []
   for metric in Cats:
        ResultsLR.append(LR_modeling(Cmodel, metric, "CHM-151", col_var))
   lr_df = pd.DataFrame(ResultsLR)
   lr_df.reset_index()
   lr_df.rename({0: "Course", 2: "Metric", 3: "RMSE", 4: "R2"}, inplace = True)
   lr_df.to_csv("LR_data.csv")
   LR_plots(lr_df, "R-Squared Values for Modeling with All Variables", "CLRfull.png")
  
       #physics
   ResultsLR = []
   for metric in Cats:
        ResultsLR.append(LR_modeling(Cmodel, metric, "PHY-151", col_var))
   lr_df = pd.DataFrame(ResultsLR)
   lr_df.reset_index()
   lr_df.rename({0: "Course", 2: "Metric", 3: "RMSE", 4: "R2"}, inplace = True)
   lr_df.to_csv("LR_data.csv")
   LR_plots(lr_df, "R-Squared Values for Modeling with All Variables", "PLRfull.png")
   
     #engineering
   ResultsLR = []
   for metric in Cats:
        ResultsLR.append(LR_modeling(Cmodel, metric, "EGR-151", col_var))
   lr_df = pd.DataFrame(ResultsLR)
   lr_df.reset_index()
   lr_df.rename({0: "Course", 2: "Metric", 3: "RMSE", 4: "R2"}, inplace = True)
   lr_df.to_csv("LR_data.csv")
   LR_plots(lr_df, "R-Squared Values for Modeling with All Variables", "ELRfull.png")
  
   print("Running KNN matching")
   
   #KNN Matching simple - Simple Match
   
   Cats = ["Pass", "Fail", "W"]
   
   print("KNN with just math grade")
     #chemistry
   ResultsKNNs = []
   for metric in Cats:
        ResultsKNNs.append(KNN_modeling(Cmodel, metric, "CHM-151", ["GradeM"], "M"))
   knn_df1s = pd.DataFrame(ResultsKNNs[0])
   knn_df2s = pd.DataFrame(ResultsKNNs[1])
   knn_df3s = pd.DataFrame(ResultsKNNs[2])
   KNN_plots(knn_df1s, "Pass", "CKNN_Pass_S.png")
   KNN_plots(knn_df2s, "Didn't Pass", "CKNN_Fail_S.png")
   KNN_plots(knn_df3s, "Withdraw", "CKNN_W_S.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("CKNN_datas.csv")

     #physics
   ResultsKNNs = []
   for metric in Cats:
        ResultsKNNs.append(KNN_modeling(Pmodel, metric, "PHY-151", ["GradeM"], "M"))
   knn_df1s = pd.DataFrame(ResultsKNNs[0])
   knn_df2s = pd.DataFrame(ResultsKNNs[1])
   knn_df3s = pd.DataFrame(ResultsKNNs[2])
   KNN_plots(knn_df1s, "Pass", "PKNN_Pass_S.png")
   KNN_plots(knn_df2s, "Didn't Pass", "PKNN_Fail_S.png")
   KNN_plots(knn_df3s, "Withdraw", "PKNN_W_S.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("PKNN_datas.csv")
   
     #engineering
   ResultsKNNs = []
   for metric in Cats:
        ResultsKNNs.append(KNN_modeling(Emodel, metric, "EGR-150", ["GradeM"], "M"))
   knn_df1s = pd.DataFrame(ResultsKNNs[0])
   knn_df2s = pd.DataFrame(ResultsKNNs[1])
   knn_df3s = pd.DataFrame(ResultsKNNs[2])
   KNN_plots(knn_df1s, "Pass", "EKNN_Pass_S.png")
   KNN_plots(knn_df2s, "Didn't Pass", "EKNN_Fail_S.png")
   KNN_plots(knn_df3s, "Withdraw", "EKNN_W_S.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("EKNN_datas.csv")
  
   #KNN Matching - Robust
   
   print("KNN with many variates")
   Cats = ["Pass", "Fail", "W"]
   
     #chemsitry---------------------------------------------------------------------------------------
   ResultsKNN = []
   for metric in Cats:
        ResultsKNN.append(KNN_modeling(Cmodel, metric, "CHM-151", col_var, "many"))
   knn_df1s = pd.DataFrame(ResultsKNN[0])
   knn_df2s = pd.DataFrame(ResultsKNN[1])
   knn_df3s = pd.DataFrame(ResultsKNN[2])
   KNN_plots(knn_df1s, "Pass", "CKNN_Pass.png")
   KNN_plots(knn_df2s, "Didn't Pass", "CKNN_Fail.png")
   KNN_plots(knn_df3s, "Withdraw", "CKNN_W.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("CKNN_data.csv")

     #physics
   ResultsKNN = []
   for metric in Cats:
        ResultsKNN.append(KNN_modeling(Pmodel, metric, "PHY-151", col_var, "many"))
   knn_df1s = pd.DataFrame(ResultsKNN[0])
   knn_df2s = pd.DataFrame(ResultsKNN[1])
   knn_df3s = pd.DataFrame(ResultsKNN[2])
   KNN_plots(knn_df1s, "Pass", "PKNN_Pass.png")
   KNN_plots(knn_df2s, "Didn't Pass", "PKNN_Fail.png")
   KNN_plots(knn_df3s, "Withdraw", "PKNN_W.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("PKNN_data.csv")
   
     #engineering
   ResultsKNN = []
   for metric in Cats:
        ResultsKNN.append(KNN_modeling(Emodel, metric, "EGR-150", col_var, "many"))
   knn_df1s = pd.DataFrame(ResultsKNN[0])
   knn_df2s = pd.DataFrame(ResultsKNN[1])
   knn_df3s = pd.DataFrame(ResultsKNN[2])
   KNN_plots(knn_df1s, "Pass", "EKNN_Pass.png")
   KNN_plots(knn_df2s, "Didn't Pass", "EKNN_Fail.png")
   KNN_plots(knn_df3s, "Withdraw", "EKNN_W.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("EKNN_data.csv")
   
      #KNN but without math grades
   
   print("KNN without math grade")
   col_var = ["Age", "dt", "GenderN", "DeliveryN"]

     #chemsitry
   ResultsKNN = []
   for metric in Cats:
        ResultsKNN.append(KNN_modeling(Cmodel, metric, "CHM-151", col_var, "SM"))
   knn_df1s = pd.DataFrame(ResultsKNN[0])
   knn_df2s = pd.DataFrame(ResultsKNN[1])
   knn_df3s = pd.DataFrame(ResultsKNN[2])
   KNN_plots(knn_df1s, "Pass", "CKNN_Passnm.png")
   KNN_plots(knn_df2s, "Didn't Pass", "CKNN_Failnm.png")
   KNN_plots(knn_df3s, "Withdraw", "CKNN_Wnm.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("CKNN_datanm.csv")

     #physics
   ResultsKNN = []
   for metric in Cats:
        ResultsKNN.append(KNN_modeling(Pmodel, metric, "PHY-151", col_var, "SM"))
   knn_df1s = pd.DataFrame(ResultsKNN[0])
   knn_df2s = pd.DataFrame(ResultsKNN[1])
   knn_df3s = pd.DataFrame(ResultsKNN[2])
   KNN_plots(knn_df1s, "Pass", "PKNN_Passnm.png")
   KNN_plots(knn_df2s, "Didn't Pass", "PKNN_Failnm.png")
   KNN_plots(knn_df3s, "Withdraw", "PKNN_Wnm.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("PKNN_datanm.csv")
   
     #engineering
   ResultsKNN = []
   for metric in Cats:
        ResultsKNN.append(KNN_modeling(Emodel, metric, "EGR-150", col_var, "SM"))
   knn_df1s = pd.DataFrame(ResultsKNN[0])
   knn_df2s = pd.DataFrame(ResultsKNN[1])
   knn_df3s = pd.DataFrame(ResultsKNN[2])
   KNN_plots(knn_df1s, "Pass", "EKNN_Passnm.png")
   KNN_plots(knn_df2s, "Didn't Pass", "EKNN_Failnm.png")
   KNN_plots(knn_df3s, "Withdraw", "EKNN_Wnm.png")
   knn_dfs = pd.concat([knn_df1s, knn_df2s, knn_df3s])
   knn_dfs.reset_index()
   knn_dfs.to_csv("EKNN_datanm.csv")
   
   return 0
   
#functions
# plot the J score for the KNN models
def KNN_plots(df, comments, filename):
     index = df[1]
     plt.clf()
     plt.plot(index, df[3])
     plt.plot(index, df[4])
     plt.xlabel("k-Value")
     plt.ylabel("Jaccard Score")
     plt.title(comments)
     plt.legend(["Jaccard with Training Data", "Jaccard with Test Data"])
     plt.savefig(filename)
     plt.ylim(bottom = 0, top = 1)
     plt.clf()
     plt.close()
     return 0

# plot the R2 for the linear models
def LR_plots(df, comments, filename):
     index = ["Passing", "Not\nPassing", "Withdrawing", "Letter\nGrade"]
     plt.bar(index, df[3])
     plt.ylabel("R-Squared")
     plt.title(comments)
     plt.savefig(filename)
     plt.clf()
     plt.close()
          
          
#KNN modeling
def KNN_modeling(df, target, course, variates, comment):
   x = df.loc[:, variates]
   y = df.loc[:, target]
   results = []
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
   for k in range(1, 30):
     neighbors = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train) 
     yhat = neighbors.predict(x_test) #makes predictions, builds model
     TrainAcc = metrics.jaccard_score(y_train, neighbors.predict(x_train)) #Jaccard score for train data
     TestAcc = metrics.jaccard_score(y_test, yhat) #Jaccard Score for test data
     DAC(y_train, neighbors.predict(x_train), k, course, "train", comment)
     DAC(y_test, neighbors.predict(x_test), k, course, "test", comment)
     results.append([course, k, target, TrainAcc, TestAcc])
   return results

#confusion matrix
def DAC(model_data, test_data, k, course, scenario, comment):
     y_true = model_data
     y_pred = test_data
     name = f"Confusion Matrix for {course} using {scenario} data k = {k}"
     cm = confusion_matrix(y_true, y_pred)
     disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Pass", "Fail"])
     disp.plot()
     filename = f"CM_{course}_{scenario}_{k}_{comment}.png"
     plt.title(name)
     plt.savefig(filename)
     plt.clf()
     plt.close()
     return 0

#linear modeling
def LR_modeling(df, target, course, variates):
     #sets up test and training data
   x = df.loc[:, variates]
   y = df.loc[:, target]
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
   
    #linear modeling
   regrC = linear_model.LinearRegression()
   regrC.fit(x_train, y_train)
   
     #model validation
   y_hat = regrC.predict(x_test) #builds model
   
     #calculates R2 and RMSE
   RMSE = np.mean((y_hat - y_test) ** 2) #returns MSE
   R2 = regrC.score(x, y) #prints variance of data
   results = [course, target, RMSE, R2]
   return results


#visualization of the correlations
def corr_plot(df, course):
    P = []
    F = []
    W = []
    G = []
    set = [4, 5, 7, 9, 10, 11, 12, 13, 14]
    for j in set:
         P.append(df.iat[1, j])
         F.append(df.iat[2, j])
         W.append(df.iat[3, j])
         G.append(df.iat[6, j])
    values = [P, F, W, G]
    index = ["Year\nof\nBirth", "Delivery\nMethod", "Gender", "Year\nand\nterm", "EHI", "Age\nin\nCourse", "Math\nGrade", "Time\nBetween\nCoures", "Del. Meth.\n(Math)"]
    cycle = ["Pass", "Did't Pass", "Withdraw", "got a letter grade in"]
    for i in range(0, 4):    
          title = f"Magnitude of Variates for students who {cycle[i]} {course}"
          plt.barh(index, values[i])
          plt.xlabel("Magnitude of Variate")
          plt.ylabel("Variates")
          plt.title(title)
          figtitle = f"Variates_{cycle[i]}_{course}.png"
          plt.savefig(figtitle)
          plt.clf()
    plt.clf()     


####pairs up the data for modeling
def coursematch(df, course):
     dfsub = df.loc[df["Course Name"] == "MAT-171"]
     dfsub = dfsub.sort_values("YT", ascending = False)
     dfC = df.loc[df["Course Name"] == course]
     dfC = dfC.sort_values("YT", ascending = False)
     SSIDs = dfsub["Student ID"]
     grade = []
     dt1 = []
     DM = []
     j = 0
     for i in SSIDs:
          df_temp = dfC.loc[dfC["Student ID"] == i]
          df_temp = df_temp.sort_values("YT", ascending = False)
          df_mtemp = dfsub.loc[dfsub["Student ID"] == i]
          df_mtemp = df_mtemp.sort_values("YT", ascending = False)
          if len(df_temp.index > 0):
               entry = pd.DataFrame(df_temp.iloc[0]).transpose()
               mentry = pd.DataFrame(df_mtemp.iloc[0]).transpose()
               if j == 0:
                    dfrun = entry
                    grade.append(mentry.iat[0, 7])
                    dt1.append(mentry.iat[0,10])
                    DM.append(mentry.iat[0,6])
                    j = j + 1
               elif j != 0:
                    dfrun = pd.concat([dfrun,entry], ignore_index = True)
                    grade.append(mentry.iat[0, 7])
                    dt1.append(mentry.iat[0,10])
                    DM.append(mentry.iat[0,6])
     dfrun["GradeM"] = grade
     dfrun["MYT"] = dt1
     dfrun["dt"] = dfrun["YT"] - dfrun["MYT"]
     dfrun["DelM"] = DM
     dfrun.drop(columns = ["MYT"], inplace = True)
     return dfrun

#####functions for transforming the raw data
#grades to numbers
def grades_to_numbers(df):
    df_temp = pd.get_dummies(df, columns = ["Grade"], dtype = float)
    df_temp["Grade"] = df["Grade"]
    df_temp["Pass"] = df_temp["Grade_A"] + df_temp["Grade_B"] + df_temp["Grade_C"]
    df_temp["GradeN"] = 5 * df_temp["Grade_A"] + 4 * df_temp["Grade_B"] + 3 * df_temp["Grade_C"] +  2 * df_temp["Grade_D"] + 1 * df_temp["Grade_F"]
    df_temp["Pass"] = df_temp["Grade_A"] + df_temp["Grade_B"] + df_temp["Grade_C"]
    df_temp["Fail"] = df_temp["Grade_D"] + 1 * df_temp["Grade_F"]
    df_temp["W"] = df_temp["Grade_W"] 
    df_temp.drop(columns = ["Grade_A", "Grade_B", "Grade_C", "Grade_D", "Grade_F", "Grade_W"], inplace = True)
    return df_temp

#delivery to numbers
def devilery_to_numbers(df):
    df = df.rename(columns = {"X Sec Delivery Method ": "Del"})
    df_temp = pd.get_dummies(df, columns = ["Del"], dtype = float)
    df_temp["Delivery"] = df["Del"]
    df_temp["DeliveryN"] = 0 * df_temp["Del_IN"] + 0 * df_temp["Del_TR"] + 0.5 * df_temp["Del_BL"] +  0.5 * df_temp["Del_HY"] + 1 * df_temp["Del_WB"]
    df_temp.drop(columns = ["Del_IN", "Del_TR", "Del_BL", "Del_HY", "Del_WB"], inplace = True)
    return df_temp

#gender to numbers
def gender_to_numbers(df):
    df_temp = pd.get_dummies(df, columns = ["Gender"], dtype = float)
    df_temp["Gender"] = df["Gender"]
    df_temp["GenderN"] = 0 * df_temp["Gender_M"] + 1 * df_temp["Gender_F"]
    df_temp.drop(columns = ["Gender_M", "Gender_F"], inplace = True)
    return df_temp

#gender to numbers
def EHI_to_numbers(df):
    df_temp = pd.get_dummies(df, columns = ["EHI"], dtype = float)
    df_temp["EHI"] = df["EHI"]
    df_temp["EHIN"] = 0 * df_temp["EHI_Low"] + 1 * df_temp["EHI_Medium-Low"] + 2 * df_temp["EHI_Medium"] + 3 * df_temp["EHI_Medium-High"] + 4 * df_temp["EHI_High"]
    df_temp.drop(columns = ["EHI_Low", "EHI_Medium-Low", "EHI_Medium", "EHI_Medium-High", "EHI_High"], inplace = True)
    return df_temp

#drop failing MAT-171 scores
 #drop C or lower math grades 
def dummy_drop(df):
    df_temp = df[df["Course Name"] == "MAT-171"]
    df_C = df[df["Course Name"] != "MAT-171"]
    df1 = df_temp[df_temp["Grade"] == "A"]
    df2 = df_temp[df_temp["Grade"] == "B"]
    df3 = df_temp[df_temp["Grade"] == "C"]
    df_Return = pd.concat([df_C, df1, df2, df3])
    return df_Return

#create year-term 
def year_term(df):
    df_temp = pd.get_dummies(df, columns = ["Term"], dtype = float)
    df_temp["Term"] = df["Term"]
    df_temp["YT"] = df["Year"].astype(float) + 0 * df_temp["Term_SP"] + 0.33 * df_temp["Term_SU"] + 0.66 * df_temp["Term_FA"]
    df_temp.drop(columns = ["Term_SP", "Term_SU", "Term_FA"], inplace = True)
    return df_temp

#####function for generating the grade distributions
#plots a grade distribution
def Grade_distr(df, course, statement, a):
    dist = df["Grade"].value_counts()
    dist2 = pd.DataFrame(dist)
    dist2.reset_index(inplace = True)
    dist2 = dist2.sort_values("index")
    title = f"Grade Distribution {course} for students {statement} MAT171"
    plt.bar(dist2["index"], dist2["Grade"]/dist2["Grade"].sum())
    plt.xlabel("Grade")
    plt.ylabel("Distribution")
    plt.title(title)
    plt.ylim(0, a)
    figtitle = f"{course}GradeDistr{statement}MAT171.png"
    plt.savefig(figtitle)
    plt.clf()

#####returning the grande transfer success
#creates pairs of students from cohort
def pairer(df, course):
    SSIDs = df["Student ID"].unique()
    array = []
    for i in SSIDs:
        dftemp = df.loc[df["Student ID"] == i]
        dftemp1 = dftemp.loc[dftemp["Course Name"] == "MAT-171"]
        dftemp1 = dftemp1.sort_values("YT", ascending = False)
        dftemp2 = dftemp.loc[dftemp["Course Name"] == course]
        dftemp2 = dftemp2.sort_values("YT", ascending = False)
        if len(dftemp1.index) > 0 and len(dftemp2.index) > 0:
            dftemp1 = dftemp1.iat[0, 5]
            dftemp2 = dftemp2.iat[0, 5]
            array.append([GradeDict[dftemp1], GradeDict[dftemp2]])
    dfreturned = pd.DataFrame(array)
    return dfreturned 

#returns probability of grade transfer
def success_numbers(df, grade):
     a = len(df[df[1] > grade].index)
     b = len(df[df[1] == grade].index)
     c = len(df[df[1] < grade].index)
     f = len(df[df[1] < 3].index)
     g = len(df[df[1] == 0].index)
     d = a + b + c
     e = d - f
     return [a/d, b/d, c/d, e/d, f/d, f, 1 - g/d, g/d]

#cycles the grade transfer
def success_generator(df, course):
     array = []
     df2 = pairer(df, course)
     for i in range(3, 6):
          df3 = df2.loc[df2[0] == i]
          output = success_numbers(df3, i)
          array.append(output)
     newarray = [[array[2][0], array[1][0], array[0][0]], [array[2][1], array[1][1], array[0][1]], [array[2][2], array[1][2], array[0][2]], [array[2][0] + array[2][1], array[1][0] + array[1][1], array[0][0] + array[0][1]]]
     N = array[0][5] + array[1][5] + array[2][5]
     grade_transfer_plot(newarray, course, N)
     success_array = [[array[2][3], array[1][3], array[0][3]], [array[2][4], array[1][4], array[0][4]]]
     success_plot(success_array, course, N)
     retention_array = [[array[2][6], array[1][6], array[0][6]], [array[2][7], array[1][7], array[0][7]]]
     retention_plot(retention_array, course, N)
     return 0
     
#plotter for the grade transfer
def grade_transfer_plot(array, course, N):
    title = f"Grade Transfer for {course}"
    x = ["A", "B", "C"]
    plt.bar(x, array[0], color = "green")
    plt.bar(x, array[1], bottom = array[0], color = "blue")
    plt.bar(x, array[2], bottom = array[3], color = "red")
    plt.legend(["Grade was higher", "Did was the same", "Grade was lower"])
    plt.xlabel("Grade in MAT-151")
    ylab = f"Grade in {course}"
    plt.ylabel(ylab)
    plt.title(title)
    figtitle = f"TS_{course}vsMAT171N{N}.png"
    plt.savefig(figtitle)
    plt.clf()
    
#plotter success rate
def success_plot(array, course, N):
    title = f"Student Passing Rates for {course}"
    x = ["A", "B", "C"]
    plt.bar(x, array[0], color = "green")
    plt.bar(x, array[1], bottom = array[0], color = "blue")
    plt.legend(["Passed the Course with C or higher", "Did not pass the course"])
    plt.xlabel("Grade in MAT-151")
    ylab = f"Percentage of Students"
    plt.ylabel(ylab)
    plt.title(title)
    figtitle = f"SS_{course}vsMAT171N{N}.png"
    plt.savefig(figtitle)
    plt.clf()

#plotter retention rate
def retention_plot(array, course, N):
    title = f"Student Retention for {course}"
    x = ["A", "B", "C"]
    plt.bar(x, array[0], color = "green")
    plt.bar(x, array[1], bottom = array[0], color = "blue")
    plt.legend(["Did not withdraw", "Withdrew"])
    plt.xlabel("Grade in MAT-151")
    ylab = f"Percentage of Students"
    plt.ylabel(ylab)
    plt.title(title)
    figtitle = f"RR_{course}vsMAT171N{N}.png"
    plt.savefig(figtitle)
    plt.clf()
         
main()
