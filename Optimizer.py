#libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

#main script
def main():
   #import data files
   
   print("So you want to ruin your computer...")
   print("Let's start by reading in the CSV files")
   dfCwith = pd.read_csv("./CHwith.csv")
   dfPwith = pd.read_csv("./PYwith.csv")
   dfEwith = pd.read_csv("./ENGwith.csv")
   dfCpred = pd.read_csv("./CModelTest.csv")
   
   print("Now let's clean the files")
      #clean datafiles
        #drop NA values
   dfCwith.dropna(inplace = True)
   dfPwith.dropna(inplace = True)
   dfEwith.dropna(inplace = True)
   
        #drops STC SEC NAME and Section columns
   dfCwith.drop(columns = ["Stc Sec Name", "Section"], inplace = True)
   dfPwith.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
   dfEwith.drop(columns = ["Stc Sec Name", "Section"], inplace = True) 
      
     #Clean up grades
   dfCwith["Grade"] = dfCwith["Grade"].str[0]
   dfPwith["Grade"] = dfPwith["Grade"].str[0]
   dfEwith["Grade"] = dfEwith["Grade"].str[0]
   
     #drops D, F, W students in MAT-171
   dfCwith = dummy_drop(dfCwith)
   dfPwith = dummy_drop(dfPwith)
   dfEwith = dummy_drop(dfEwith)

      #grades to numbers 
   dfCwith = grades_to_numbers(dfCwith)
   dfPwith = grades_to_numbers(dfPwith)
   dfEwith = grades_to_numbers(dfEwith)
   
        #converts delivery method to numbers
   dfCwith = devilery_to_numbers(dfCwith)
   dfPwith = devilery_to_numbers(dfPwith)
   dfEwith = devilery_to_numbers(dfEwith)
   
        #convers gender to numbers
   dfCwith = gender_to_numbers(dfCwith)
   dfPwith = gender_to_numbers(dfPwith)
   dfEwith = gender_to_numbers(dfEwith)

        #convers EHI to numbers
   dfCwith = EHI_to_numbers(dfCwith)
   dfPwith = EHI_to_numbers(dfPwith)
   dfEwith = EHI_to_numbers(dfEwith)
   
        #creates year and term columns
   dfCwith["Year"] = dfCwith["Stc Term"].str[0:4].astype(int)
   dfCwith["Term"] = dfCwith["Stc Term"].str[4:6]
   dfPwith["Year"] = dfPwith["Stc Term"].str[0:4].astype(int)
   dfPwith["Term"] = dfPwith["Stc Term"].str[4:6]
   dfEwith["Year"] = dfEwith["Stc Term"].str[0:4].astype(int)
   dfEwith["Term"] = dfEwith["Stc Term"].str[4:6]
   
        #creates numberic year-term for analysis
   dfCwith = year_term(dfCwith)
   dfPwith = year_term(dfPwith)
   dfEwith = year_term(dfEwith)
   
        #convert objects to integers
   dfCwith["Year"] = dfCwith["Year"].astype(str).astype(int)
   dfPwith["Year"] = dfPwith["Year"].astype(str).astype(int)
   dfEwith["Year"] = dfEwith["Year"].astype(str).astype(int)
   
        #creates age column
   dfCwith["Age"] = dfCwith["Year"] - dfCwith["Year of Birth"].astype(int)
   dfPwith["Age"] = dfPwith["Year"] - dfPwith["Year of Birth"].astype(int)
   dfEwith["Age"] = dfEwith["Year"] - dfEwith["Year of Birth"].astype(int)
     
   print("Okay, I just cleaned the data, now we can get to work with modeling")

    #####modeling
         #create model sets
   col_list = ["Student ID", "Course Name", "Pass", "Fail", "W", "Year of Birth", "DeliveryN", "GradeN", "GenderN", "Year", "YT", "EHIN", "Age"]
   Cmodel = dfCwith.loc[:, col_list]
   Pmodel = dfPwith.loc[:, col_list]
   Emodel = dfEwith.loc[:, col_list]

   #pairs up last math grade with last course grade
   Cmodel = coursematch(Cmodel, "CHM-151")
   Pmodel = coursematch(Pmodel, "PHY-151")
   Emodel = coursematch(Emodel, "EGR-150")
   
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


   print("Now lets normalize the models")

#    #normalize data sets (min-max)
   norm_var = ["Year of Birth", "GradeM", "GradeN", "Year", "YT", "EHIN", "Age", "dt"]
   for column in norm_var:
        Cmodel[column] = (Cmodel[column] - Cmodel[column].min()) / (Cmodel[column].max() - Cmodel[column].min())
   for column in norm_var:
     Pmodel[column] = (Pmodel[column] - Pmodel[column].min()) / (Pmodel[column].max() - Pmodel[column].min())
   for column in norm_var:
        Emodel[column] = (Emodel[column] - Emodel[column].min()) / (Emodel[column].max() - Emodel[column].min())
   
   #save model data
   Cmodel.to_csv("CmodelOpt.csv")   
   Pmodel.to_csv("PmodelOpt.csv")   
   Emodel.to_csv("EmodelOpt.csv")
     
   #correlation variables for advanced modeling, with least important variates listed for
   col_varC = ["GradeM", "dt", "DeliveryN", "DelM", "Age", "Year of Birth", "EHIN", "Pass", "Fail", "W", "Student ID", "Course Name", "Year of Birth"]
   col_varP = ["GradeM", "dt", "DeliveryN", "EHIN", "Age", "DelM", "Year of Birth", "Pass", "Fail", "W", "Student ID", "Course Name", "Year of Birth"]
   col_varE = ["GradeM", "DeliveryN", "DelM", "EHIN", "dt", "Age", "Year of Birth", "Pass", "Fail", "W", "Student ID", "Course Name", "Year of Birth"]
   print("Running KNN matching")
   
   #KNN Matching - Robust
   
   print("KNN model building")
   Metric = "Pass"
   
     #chemsitry---------------------------------------------------------------------------------------
   kf = 6 
   ifinal = 8
   
   Results = []
   for i in range(1, ifinal):
        Clist = col_varC[0:i] 
        for k in range(1, kf):
             output = KNN_modeling(Cmodel, Metric,  "CHM-151", Clist, k, i)
             Results.append(output)
   CDF = pd.DataFrame(Results)
   CDF.rename(columns = {0 : "k", 1: "Variates", 2: "Training Sensitivity", 3: "Training Specificity", 4: "Training Accuracy", 5: "Training F1", 6: "Testing Sensitivity", 7: "Testing Specificity", 8: "Testing Accuracy", 9: "Testing F1"}, inplace = True)
   results_plotter(kf, CDF, "CHM-151")         

   Results = []
   for i in range(1, ifinal): 
        Plist = col_varP[0:i] 
        for k in range(1, kf): 
             output = KNN_modeling(Cmodel, Metric,  "PHY-151", Plist, k, i)
             Results.append(output)
   PDF = pd.DataFrame(Results)
   PDF.rename(columns = {0 : "k", 1: "Variates", 2: "Training Sensitivity", 3: "Training Specificity", 4: "Training Accuracy", 5: "Training F1", 6: "Testing Sensitivity", 7: "Testing Specificity", 8: "Testing Accuracy", 9: "Testing F1"}, inplace = True)
   results_plotter(kf, PDF, "PHY-151")    

   Results = []
   for i in range(1, ifinal): #change back to 8
        Elist = col_varE[0:i] 
        for k in range(1, kf): #change back to 6
             output = KNN_modeling(Cmodel, Metric,  "EGR-150", Elist, k, i)
             Results.append(output)
   EDF = pd.DataFrame(Results)
   EDF.rename(columns = {0 : "k", 1: "Variates", 2: "Training Sensitivity", 3: "Training Specificity", 4: "Training Accuracy", 5: "Training F1", 6: "Testing Sensitivity", 7: "Testing Specificity", 8: "Testing Accuracy", 9: "Testing F1"}, inplace = True)
   results_plotter(kf, CDF, "EGR-150")    
             
    #create a new KNN model - for making predictions
#    knn_cv = KNeighborsClassifier(n_neighbors = 1).fit(X, Y)
#    predict = knn_cv.predict(x_test)
#    Student_list  = pd.DataFrame(dfCpred["Student ID"])
#    Student_list["Pass/Fail"] = predict
#    Students = Student_list[Student_list["Pass/Fail"] == 0]
#    Students = Students.drop(columns = ["Pass/Fail"])
#    Students.reset_index(inplace = True)
#    Students.drop(columns = ["index"], inplace = True)
#    Students.to_csv("StudentList.csv")
     #physics
   
     #engineering
   
   return 0
   
#functions
#creates plot of results based on k
def results_plotter(kf, results, course):
     for k in range(1, kf):
          submatrix = results[results["k"] == k]
          title = f"Metrics verses number of variates for {course} Training Data"
          plt.plot(submatrix["Variates"], submatrix["Training Sensitivity"])
          plt.plot(submatrix["Variates"], submatrix["Training Specificity"])
          plt.plot(submatrix["Variates"], submatrix["Training Accuracy"])
          plt.plot(submatrix["Variates"], submatrix["Training F1"])
          plt.xlabel("Number of variates")
          plt.ylim(0, 1)
          plt.ylabel("Metric Value")
          plt.legend(["Sensivity", "Specificity", "Accuracy", "F1"])
          plt.title(title)
          figtitle = f"Metrics_k{k}_{course}_training.png"
          plt.savefig(figtitle)
          plt.clf()
          title = f"Metrics verses number of variates for {course} Testing Data"
          plt.plot(submatrix["Variates"], submatrix["Testing Sensitivity"])
          plt.plot(submatrix["Variates"], submatrix["Testing Specificity"])
          plt.plot(submatrix["Variates"], submatrix["Testing Accuracy"])
          plt.plot(submatrix["Variates"], submatrix["Testing F1"])
          plt.xlabel("Number of variates")
          plt.ylabel("Metric Value")
          plt.ylim(0, 1)
          plt.title(title)
          plt.legend(["Sensivity", "Specificity", "Accuracy", "F1"])
          figtitle = f"Metrics_k{k}_{course}_testing.png"
          plt.savefig(figtitle)
          plt.clf()

#KNN modeling
def KNN_modeling(df, target, course, variates, k, i):
   x = df.loc[:, variates]
   y = df.loc[:, target]
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
   neighbors = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train) 
   train = DAC(y_train, neighbors.predict(x_train), k, course, "train", i)
   test = DAC(y_test, neighbors.predict(x_test), k, course, "test", i)
   results = [k, 8 - i] + train + test
   return results

#confusion matrix
def DAC(model_data, test_data, k, course, scenario, i):
     y_true = model_data
     y_pred = test_data
     name = f"Confusion Matrix for {course} using {scenario} data k = {k} and {i + 1} variates"
     cm = confusion_matrix(y_true, y_pred, labels = [1, 0])
     results = [cm[0][0]/(cm[0][0] + cm[0][1]), cm[1][1]/(cm[1][0] + cm[1][1]), (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[0][1]), 2 * cm[0][0]/(2 * cm[0][0] + cm[1][0] + cm[0][1])]
     disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Pass", "Fail"])
     disp.plot()
     filename = f"CM_{course}_{scenario}_k{k}_i{i + 1}.png"
     plt.title(name)
     plt.savefig(filename)
     plt.clf()
     plt.close()
     return results

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

#EHI to numbers
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
         
main()
