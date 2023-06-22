#libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#global variables
TermDict = {"FA": 0.7, "SP": 0, "SU": 0.3}
group_names = ["<18", "18-22", "23-26", "27-30", "31-34", "35-38", ">38"]
bins = [0, 18, 23, 27, 31, 35, 38, 100]

#main script
def main():
   #import data files
   
   print("So you want to ruin your computer...")
   print("Let's start by reading in the CSV files")
   dfCwith = pd.read_csv("./CHwith.csv")
   dfPwith = pd.read_csv("./PYwith.csv")
   dfEwith = pd.read_csv("./ENGwith.csv")
   
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
     
        #standardizes the delivery method
   dfCwith = dfCwith.replace("BL", "HY").replace("TR", "IN").replace("WB", "ON")
   dfPwith = dfPwith.replace("BL", "HY").replace("TR", "IN").replace("WB", "ON")
   dfEwith = dfEwith.replace("BL", "HY").replace("TR", "IN").replace("WB", "ON")
   
        #creates year and term columns
   dfCwith["Year"] = dfCwith["Stc Term"].str[0:4].astype(int)
   dfCwith["Term"] = dfCwith["Stc Term"].str[4:6]
   dfPwith["Year"] = dfPwith["Stc Term"].str[0:4].astype(int)
   dfPwith["Term"] = dfPwith["Stc Term"].str[4:6]
   dfEwith["Year"] = dfEwith["Stc Term"].str[0:4].astype(int)
   dfEwith["Term"] = dfEwith["Stc Term"].str[4:6]

        #convert objects to integers
   dfCwith["Year"] = dfCwith["Year"].astype(str).astype(int)
   dfPwith["Year"] = dfPwith["Year"].astype(str).astype(int)
   dfEwith["Year"] = dfEwith["Year"].astype(str).astype(int)

        #converts term and year to number
   dfCwith = year_term(dfCwith)
   dfPwith = year_term(dfPwith)
   dfEwith = year_term(dfEwith)

        #creates age column
   dfCwith["Age"] = dfCwith["Year"] - dfCwith["Year of Birth"].astype(int)
   dfPwith["Age"] = dfPwith["Year"] - dfPwith["Year of Birth"].astype(int)
   dfEwith["Age"] = dfEwith["Year"] - dfEwith["Year of Birth"].astype(int)

        #bin age ranges
   bins = [0, 18, 23, 27, 31, 35, 38, 100]
   dfCwith["AgeBin"] = pd.cut(dfCwith["Age"], bins = bins, labels = group_names)
   dfPwith["AgeBin"] = pd.cut(dfPwith["Age"], bins = bins, labels = group_names)
   dfEwith["AgeBin"] = pd.cut(dfEwith["Age"], bins = bins, labels = group_names)
   
        #clean up the formatting   
   dict_type = {"Year of Birth": float, "Year": float, "YT": float, "Age": float}
   dfCwith = dfCwith.convert_dtypes(dict_type)
   dfPwith = dfPwith.convert_dtypes(dict_type)
   dfEwith = dfEwith.convert_dtypes(dict_type)
   
     #get rid of people who failed MAT-171
   dfCwith = dummy_drop(dfCwith, "CHM-151")
   dfPwith = dummy_drop(dfPwith, "PHY-151")
   dfEwith = dummy_drop(dfEwith, "EGR-150")
      
        #grade match
   print("Let's pair up math grades")
   dfCwith = coursematch(dfCwith, "CHM-151")   
   dfPwith = coursematch(dfPwith, "PHY-151")   
   dfEwith = coursematch(dfEwith, "EGR-150")   

    #drops nonesense dt columns
   dfCwith = dfCwith[dfCwith["Time Between Courses"] >= 0 ]
   dfPwith = dfPwith[dfPwith["Time Between Courses"] >= 0 ]
   dfEwith = dfEwith[dfEwith["Time Between Courses"] >= 0 ]

    #scrubs PII
   dfCwith.drop(columns = ["Student ID", "Stc Term"], inplace = True)
   dfPwith.drop(columns = ["Student ID", "Stc Term"], inplace = True)
   dfEwith.drop(columns = ["Student ID", "Stc Term"], inplace = True)
   
     #write to CSV
   print("Writing CSVs")
   dfCwith.to_csv("Cdashboard.csv")
   dfPwith.to_csv("Pdashboard.csv")
   dfEwith.to_csv("Edashboard.csv")
   merged = pd.concat([dfCwith, dfPwith, dfEwith])
   merged.reset_index()
   merged.to_csv("MergedDeck.csv")   
      
   return 0

#functions
####pairs up the data for modeling
def coursematch(df, course):
     dfsub = df.loc[df["Course Name"] == "MAT-171"]
     dfsub = dfsub.sort_values("YT", ascending = False)
     dfsub["Math Modality"] = dfsub["X Sec Delivery Method "]
     dfsub["Math Year"] = dfsub["YT"]
     dfsub.drop([dfsub.columns[4], dfsub.columns[11]], axis = 1, inplace = True)
     dfC = df.loc[df["Course Name"] == course]
     dfC = dfC.sort_values("YT", ascending = False)
     dfC["Course Modality"] = dfC["X Sec Delivery Method "]
     dfC["Course Year"] = dfC["YT"]
     dfC.drop([dfC.columns[4], dfC.columns[11]], axis = 1, inplace = True)
     SSIDs = dfsub["Student ID"]
     grade = []
     dt1 = []
     DM = []
     j = 0
     for i in SSIDs:
          df_temp = dfC.loc[dfC["Student ID"] == i]
          df_temp = df_temp.sort_values("Course Year", ascending = False)
          df_mtemp = dfsub.loc[dfsub["Student ID"] == i]
          df_mtemp = df_mtemp.sort_values("Math Year", ascending = False)
          if len(df_temp.index > 0):
               Centry = pd.DataFrame(df_temp.iloc[0]).transpose()
               Mentry = pd.DataFrame(df_mtemp.iloc[0]).transpose()
               if j == 0:
                    dfrun = Centry
                    grade.append(Mentry.iat[0, 3])
                    dt1.append(Mentry.iat[0,13])
                    DM.append(Mentry.iat[0,12])
                    j = j + 1
               elif j != 0:
                    dfrun = pd.concat([dfrun, Centry], ignore_index = True)
                    grade.append(Mentry.iat[0, 3])
                    dt1.append(Mentry.iat[0,13])
                    DM.append(Mentry.iat[0,12])
     dfrun["Math Grade"] = grade
     dfrun["MYT"] = dt1
     dfrun["Time Between Courses"] = dfrun["Course Year"] - dfrun["MYT"]
     dfrun["Math Delivery"] = DM
     dfrun.drop(columns = ["MYT"], inplace = True)
     return dfrun

 
 #create year-term 
def year_term(df):
    df_temp = pd.get_dummies(df, columns = ["Term"], dtype = float)
    df_temp["Term"] = df["Term"]
    df_temp["YT"] = df["Year"].astype(float) + 0 * df_temp["Term_SP"] + 0.33 * df_temp["Term_SU"] + 0.66 * df_temp["Term_FA"]
    df_temp.drop(columns = ["Term_SP", "Term_SU", "Term_FA"], inplace = True)
    return df_temp

 #drop C or lower math grades 
def dummy_drop(df):
    df_temp = df[df["Course Name"] == "MAT-171"]
    df_C = df[df["Course Name"] != "MAT-171"]
    df1 = df_temp[df_temp["Grade"] == "A"]
    df2 = df_temp[df_temp["Grade"] == "B"]
    df3 = df_temp[df_temp["Grade"] == "C"]
    df_Return = pd.concat([df_C, df1, df2, df3])
    return df_Return


main()