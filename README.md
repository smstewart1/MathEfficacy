# MathEfficacy
In this project I tried to build a model to determine the likelihood of success of a student in introductory chemistry, phyics, and engineering courses.  Variates included age, time between the math course and the course of interest, economic hardness index (EHI), and gender.  The larger goal was to identify high risk students before they enter the next course so that interventions can be begin before the student begins to fails.

Initially linear models were used, but as they were insufficient, KNN models were optized. From the study, it was found the k = 1 worked the best due to how quickly neighbors began to reflect the global average

Code was run in python and used the following packages
  pandas
  scikitlearn
  matplotlib
  numpy
  seaborn

A separate file was created to generate data sets for a visualization dashboard.  

Raw data files are omitted as they contain PII.

A link to visualizations of the data can be found at

https://public.tableau.com/app/profile/stephen.stewart4198/viz/shared/7JZTRW38T

Some of the key findings, summarized better in the PowerPoint deck were the strengths of different correlations

![image](https://github.com/smstewart1/MathEfficacy/assets/107202785/3f1a64b5-3882-4cbc-9ccf-25147cd58184)

Effect of nearest neighbors on model accuracy

![image](https://github.com/smstewart1/MathEfficacy/assets/107202785/c3612f66-7ca4-4e41-8e13-fcd95d9ac5cf)

And the confusion matrices of the different KNN models

![image](https://github.com/smstewart1/MathEfficacy/assets/107202785/25cacb64-9407-4596-9654-fb7ff8e955d5)

Models with accuracies as high as 90% were developed, with good selectivity and specificity.
