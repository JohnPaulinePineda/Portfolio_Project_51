***
# Supervised Learning : Modelling Right-Censored Survival Time and Status Responses for Prediction

***
### John Pauline Pineda <br> <br> *June 23, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Predictive Model Development](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Cox Proportional Hazards Regression](#1.6.2)
        * [1.6.3 Penalized Cox Regression](#1.6.3)
        * [1.6.4 Survival Support Vector Machine](#1.6.4)
        * [1.6.5 Survival Tree](#1.6.5)
        * [1.6.6 Random Survival Forest](#1.6.6)
        * [1.6.7 Extra Survival Forest](#1.6.7)
        * [1.6.8 Gradient Boosted Survival](#1.6.8)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project implements the **Cox Proportional Hazards**, **Random Survival Forest**, **Fast Survival Support Vector Machine** and **Gradient Boosting Survival Analysis**   algorithms using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to estimate the survival probabilities of right-censored survival time and status responses. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power using the Harrel's concordance index metric. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Survival Analysis](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) deals with the analysis of time-to-event data. It focuses on the expected duration of time until one or more events of interest occur, such as death, failure, or relapse. This type of analysis is used to study and model the time until the occurrence of an event, taking into account that the event might not have occurred for all subjects during the study period. Several key aspects of survival analysis include the survival function which refers to the probability that an individual survives longer than a certain time, hazard function which describes the instantaneous rate at which events occur, given no prior event, and censoring pertaining to a condition where the event of interest has not occurred for some subjects during the observation period.

[Right-Censored Survival Data](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) occurs when the event of interest has not happened for some subjects by the end of the study period or the last follow-up time. This type of censoring is common in survival analysis because not all individuals may experience the event before the study ends, or they might drop out or be lost to follow-up. Right-censored data is crucial in survival analysis as it allows the inclusion of all subjects in the analysis, providing more accurate and reliable estimates.

[Survival Models](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) refer to statistical methods used to analyze survival data, accounting for censored observations. These models aim to describe the relationship between survival time and one or more predictor variables, and to estimate the survival function and hazard function. Survival models are essential for understanding the factors that influence time-to-event data, allowing for predictions and comparisons between different groups or treatment effects. They are widely used in clinical trials, reliability engineering, and other areas where time-to-event data is prevalent.

## 1.1. Data Background <a class="anchor" id="1.1"></a>

An open [Liver Cirrhosis Dataset](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Arjun Bhaybhang](https://www.kaggle.com/arjunbhaybhang)) was used for the analysis as consolidated from the following primary sources: 
1. Reference Book entitled **Counting Processes and Survival Analysis** from [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118150672)
2. Research Paper entitled **Efficacy of Liver Transplantation in Patients with Primary Biliary Cirrhosis** from the [New England Journal of Medicine](https://www.nejm.org/doi/abs/10.1056/NEJM198906293202602)
3. Research Paper entitled **Prognosis in Primary Biliary Cirrhosis: Model for Decision Making** from the [Hepatology](https://aasldpubs.onlinelibrary.wiley.com/doi/10.1002/hep.1840100102)

This study hypothesized that the evaluated drug, liver profile test biomarkers and various clinicopathological characteristics influence liver cirrhosis survival between patients.

The target status and survival duration variables for the study are:
* <span style="color: #FF0000">Status</span> - Status of the patient (C, censored | CL, censored due to liver transplant | D, death)
* <span style="color: #FF0000">N_Days</span> - Number of days between registration and the earlier of death, transplantation, or study analysis time (1986)

The predictor variables for the study are:
* <span style="color: #FF0000">Drug</span> - Type of administered drug to the patient (D-Penicillamine | Placebo)
* <span style="color: #FF0000">Age</span> - Patient's age (Days)
* <span style="color: #FF0000">Sex</span> - Patient's sex (Male | Female)
* <span style="color: #FF0000">Ascites</span> - Presence of ascites (Yes | No)
* <span style="color: #FF0000">Hepatomegaly</span> - Presence of hepatomegaly (Yes | No)
* <span style="color: #FF0000">Spiders</span> - Presence of spiders (Yes | No)
* <span style="color: #FF0000">Edema</span> - Presence of edema ( N, No edema and no diuretic therapy for edema | S, Edema present without diuretics or edema resolved by diuretics) | Y, Edema despite diuretic therapy)
* <span style="color: #FF0000">Bilirubin</span> - Serum bilirubin (mg/dl)
* <span style="color: #FF0000">Cholesterol</span> - Serum cholesterol (mg/dl)
* <span style="color: #FF0000">Albumin</span> - Albumin (gm/dl)
* <span style="color: #FF0000">Copper</span> - Urine copper (ug/day)
* <span style="color: #FF0000">Alk_Phos</span> - Alkaline phosphatase (U/liter)
* <span style="color: #FF0000">SGOT</span> - Serum glutamic-oxaloacetic transaminase (U/ml)
* <span style="color: #FF0000">Triglycerides</span> - Triglicerides (mg/dl)
* <span style="color: #FF0000">Platelets</span> - Platelets (cubic ml/1000)
* <span style="color: #FF0000">Prothrombin</span> - Prothrombin time (seconds)
* <span style="color: #FF0000">Stage</span> - Histologic stage of disease (Stage I | Stage II | Stage III | Stage IV)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **418 rows** (observations)
    * **20 columns** (variables)
        * **1/20 metadata** (object)
            * <span style="color: #FF0000">ID</span>
        * **2/20 target | duration** (categorical | numeric)
             * <span style="color: #FF0000">Status</span>
             * <span style="color: #FF0000">N_Days</span>
        * **10/20 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
             * <span style="color: #FF0000">Bilirubin</span>
             * <span style="color: #FF0000">Cholesterol</span>
             * <span style="color: #FF0000">Albumin</span>
             * <span style="color: #FF0000">Copper</span>
             * <span style="color: #FF0000">Alk_Phos</span>
             * <span style="color: #FF0000">SGOT</span>
             * <span style="color: #FF0000">Triglycerides</span>
             * <span style="color: #FF0000">Platelets</span>
             * <span style="color: #FF0000">Prothrombin</span>
        * **7/20 predictor** (categorical)
             * <span style="color: #FF0000">Drug</span>
             * <span style="color: #FF0000">Sex</span>
             * <span style="color: #FF0000">Ascites</span>
             * <span style="color: #FF0000">Hepatomegaly</span>
             * <span style="color: #FF0000">Spiders</span>
             * <span style="color: #FF0000">Edema</span>
             * <span style="color: #FF0000">Stage</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats
```


```python
##################################
# Loading the dataset
##################################
cirrhosis_survival = pd.read_csv('Cirrhosis_Survival.csv')
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival.shape)
```

    Dataset Dimensions: 
    


    (418, 20)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cirrhosis_survival.dtypes)
```

    Column Names and Data Types:
    


    ID                 int64
    N_Days             int64
    Status            object
    Drug              object
    Age                int64
    Sex               object
    Ascites           object
    Hepatomegaly      object
    Spiders           object
    Edema             object
    Bilirubin        float64
    Cholesterol      float64
    Albumin          float64
    Copper           float64
    Alk_Phos         float64
    SGOT             float64
    Tryglicerides    float64
    Platelets        float64
    Prothrombin      float64
    Stage            float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cirrhosis_survival.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>N_Days</th>
      <th>Status</th>
      <th>Drug</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>400</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>21464</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>14.5</td>
      <td>261.0</td>
      <td>2.60</td>
      <td>156.0</td>
      <td>1718.0</td>
      <td>137.95</td>
      <td>172.0</td>
      <td>190.0</td>
      <td>12.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4500</td>
      <td>C</td>
      <td>D-penicillamine</td>
      <td>20617</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>1.1</td>
      <td>302.0</td>
      <td>4.14</td>
      <td>54.0</td>
      <td>7394.8</td>
      <td>113.52</td>
      <td>88.0</td>
      <td>221.0</td>
      <td>10.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1012</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>25594</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>S</td>
      <td>1.4</td>
      <td>176.0</td>
      <td>3.48</td>
      <td>210.0</td>
      <td>516.0</td>
      <td>96.10</td>
      <td>55.0</td>
      <td>151.0</td>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1925</td>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>19994</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>S</td>
      <td>1.8</td>
      <td>244.0</td>
      <td>2.54</td>
      <td>64.0</td>
      <td>6121.8</td>
      <td>60.63</td>
      <td>92.0</td>
      <td>183.0</td>
      <td>10.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1504</td>
      <td>CL</td>
      <td>Placebo</td>
      <td>13918</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>3.4</td>
      <td>279.0</td>
      <td>3.53</td>
      <td>143.0</td>
      <td>671.0</td>
      <td>113.15</td>
      <td>72.0</td>
      <td>136.0</td>
      <td>10.9</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Taking the ID column as the index
##################################
cirrhosis_survival.set_index(['ID'], inplace=True)
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cirrhosis_survival.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N_Days</th>
      <td>418.0</td>
      <td>1917.782297</td>
      <td>1104.672992</td>
      <td>41.00</td>
      <td>1092.7500</td>
      <td>1730.00</td>
      <td>2613.50</td>
      <td>4795.00</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>418.0</td>
      <td>18533.351675</td>
      <td>3815.845055</td>
      <td>9598.00</td>
      <td>15644.5000</td>
      <td>18628.00</td>
      <td>21272.50</td>
      <td>28650.00</td>
    </tr>
    <tr>
      <th>Bilirubin</th>
      <td>418.0</td>
      <td>3.220813</td>
      <td>4.407506</td>
      <td>0.30</td>
      <td>0.8000</td>
      <td>1.40</td>
      <td>3.40</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>284.0</td>
      <td>369.510563</td>
      <td>231.944545</td>
      <td>120.00</td>
      <td>249.5000</td>
      <td>309.50</td>
      <td>400.00</td>
      <td>1775.00</td>
    </tr>
    <tr>
      <th>Albumin</th>
      <td>418.0</td>
      <td>3.497440</td>
      <td>0.424972</td>
      <td>1.96</td>
      <td>3.2425</td>
      <td>3.53</td>
      <td>3.77</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>Copper</th>
      <td>310.0</td>
      <td>97.648387</td>
      <td>85.613920</td>
      <td>4.00</td>
      <td>41.2500</td>
      <td>73.00</td>
      <td>123.00</td>
      <td>588.00</td>
    </tr>
    <tr>
      <th>Alk_Phos</th>
      <td>312.0</td>
      <td>1982.655769</td>
      <td>2140.388824</td>
      <td>289.00</td>
      <td>871.5000</td>
      <td>1259.00</td>
      <td>1980.00</td>
      <td>13862.40</td>
    </tr>
    <tr>
      <th>SGOT</th>
      <td>312.0</td>
      <td>122.556346</td>
      <td>56.699525</td>
      <td>26.35</td>
      <td>80.6000</td>
      <td>114.70</td>
      <td>151.90</td>
      <td>457.25</td>
    </tr>
    <tr>
      <th>Tryglicerides</th>
      <td>282.0</td>
      <td>124.702128</td>
      <td>65.148639</td>
      <td>33.00</td>
      <td>84.2500</td>
      <td>108.00</td>
      <td>151.00</td>
      <td>598.00</td>
    </tr>
    <tr>
      <th>Platelets</th>
      <td>407.0</td>
      <td>257.024570</td>
      <td>98.325585</td>
      <td>62.00</td>
      <td>188.5000</td>
      <td>251.00</td>
      <td>318.00</td>
      <td>721.00</td>
    </tr>
    <tr>
      <th>Prothrombin</th>
      <td>416.0</td>
      <td>10.731731</td>
      <td>1.022000</td>
      <td>9.00</td>
      <td>10.0000</td>
      <td>10.60</td>
      <td>11.10</td>
      <td>18.00</td>
    </tr>
    <tr>
      <th>Stage</th>
      <td>412.0</td>
      <td>3.024272</td>
      <td>0.882042</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>4.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variables
##################################
print('Categorical Variable Summary:')
display(cirrhosis_survival.describe(include='object').transpose())
```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Status</th>
      <td>418</td>
      <td>3</td>
      <td>C</td>
      <td>232</td>
    </tr>
    <tr>
      <th>Drug</th>
      <td>312</td>
      <td>2</td>
      <td>D-penicillamine</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>418</td>
      <td>2</td>
      <td>F</td>
      <td>374</td>
    </tr>
    <tr>
      <th>Ascites</th>
      <td>312</td>
      <td>2</td>
      <td>N</td>
      <td>288</td>
    </tr>
    <tr>
      <th>Hepatomegaly</th>
      <td>312</td>
      <td>2</td>
      <td>Y</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Spiders</th>
      <td>312</td>
      <td>2</td>
      <td>N</td>
      <td>222</td>
    </tr>
    <tr>
      <th>Edema</th>
      <td>418</td>
      <td>3</td>
      <td>N</td>
      <td>354</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 12 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">Tryglicerides</span>: Null.Count = 136, Fill.Rate = 0.675
    * <span style="color: #FF0000">Cholesterol</span>: Null.Count = 134, Fill.Rate = 0.679
    * <span style="color: #FF0000">Copper</span>: Null.Count = 108, Fill.Rate = 0.741
    * <span style="color: #FF0000">Drug</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Ascites</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Hepatomegaly</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Spiders</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Alk_Phos</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">SGOT</span>: Null.Count = 106, Fill.Rate = 0.746
    * <span style="color: #FF0000">Platelets</span>: Null.Count = 11, Fill.Rate = 0.974
    * <span style="color: #FF0000">Stage</span>: Null.Count = 6, Fill.Rate = 0.986
    * <span style="color: #FF0000">Prothrombin</span>: Null.Count = 2, Fill.Rate = 0.995
3. 142 observations noted with at least 1 missing data. From this number, 14 observations reported high Missing.Rate>0.2.
    * 91 Observations: Missing.Rate = 0.450 (9 columns)
    * 15 Observations: Missing.Rate = 0.500 (10 columns)
    * 28 Observations: Missing.Rate = 0.100 (2 columns)
    * 8 Observations: Missing.Rate = 0.050 (1 column)
4. Low variance observed for 3 variables with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">Ascites</span>: First.Second.Mode.Ratio = 12.000
    * <span style="color: #FF0000">Sex</span>: First.Second.Mode.Ratio = 8.500
    * <span style="color: #FF0000">Edema</span>: First.Second.Mode.Ratio = 8.045
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High and marginally high skewness observed for 2 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">Cholesterol</span>: Skewness = +3.409
    * <span style="color: #FF0000">Alk_Phos</span>: Skewness = +2.993


```python
##################################
# Counting the number of duplicated rows
##################################
cirrhosis_survival.duplicated().sum()
```




    0




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cirrhosis_survival.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cirrhosis_survival.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cirrhosis_survival)] * len(cirrhosis_survival.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cirrhosis_survival.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cirrhosis_survival.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>float64</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
print('Number of Columns with Missing Data:', str(len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])))
```

    Number of Columns with Missing Data: 12
    


```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
print('Columns with Missing Data:')
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
```

    Columns with Missing Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>float64</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1.00)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cirrhosis_survival.index.values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cirrhosis_survival.columns)] * len(cirrhosis_survival))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cirrhosis_survival.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Exploring the rows
# for missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>19</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Fill.Rate < 1.00
##################################
print('Number of Rows with Missing Data:',str(len(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])))
```

    Number of Rows with Missing Data: 142
    


```python
##################################
# Identifying the rows
# with Fill.Rate < 1.00
##################################
print('Rows with Missing Data:')
display(all_row_quality_summary[all_row_quality_summary['Missing.Rate']>0])
```

    Rows with Missing Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>19</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>19</td>
      <td>2</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>19</td>
      <td>9</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# based on different Fill.Rate categories
##################################
missing_rate_categories = all_row_quality_summary['Missing.Rate'].value_counts().reset_index()
missing_rate_categories.columns = ['Missing.Rate.Category','Missing.Rate.Count']
display(missing_rate_categories.sort_values(['Missing.Rate.Category'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing.Rate.Category</th>
      <th>Missing.Rate.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.526316</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.473684</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.105263</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.052632</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>276</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Missing.Rate > 0.40
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.40)]
```


```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cirrhosis_survival_numeric = cirrhosis_survival.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cirrhosis_survival_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cirrhosis_survival_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cirrhosis_survival_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cirrhosis_survival_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cirrhosis_survival_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0] for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1] for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cirrhosis_survival_numeric[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cirrhosis_survival_numeric[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cirrhosis_survival_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cirrhosis_survival_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cirrhosis_survival_numeric)] * len(cirrhosis_survival_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cirrhosis_survival_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cirrhosis_survival_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>41.00</td>
      <td>1917.782297</td>
      <td>1730.00</td>
      <td>4795.00</td>
      <td>1434.00</td>
      <td>3445.00</td>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>399</td>
      <td>418</td>
      <td>0.954545</td>
      <td>0.472602</td>
      <td>-0.482139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>9598.00</td>
      <td>18533.351675</td>
      <td>18628.00</td>
      <td>28650.00</td>
      <td>19724.00</td>
      <td>18993.00</td>
      <td>7</td>
      <td>6</td>
      <td>1.166667</td>
      <td>344</td>
      <td>418</td>
      <td>0.822967</td>
      <td>0.086850</td>
      <td>-0.616730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bilirubin</td>
      <td>0.30</td>
      <td>3.220813</td>
      <td>1.40</td>
      <td>28.00</td>
      <td>0.70</td>
      <td>0.60</td>
      <td>33</td>
      <td>31</td>
      <td>1.064516</td>
      <td>98</td>
      <td>418</td>
      <td>0.234450</td>
      <td>2.717611</td>
      <td>8.065336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cholesterol</td>
      <td>120.00</td>
      <td>369.510563</td>
      <td>309.50</td>
      <td>1775.00</td>
      <td>260.00</td>
      <td>316.00</td>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
      <td>201</td>
      <td>418</td>
      <td>0.480861</td>
      <td>3.408526</td>
      <td>14.337870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albumin</td>
      <td>1.96</td>
      <td>3.497440</td>
      <td>3.53</td>
      <td>4.64</td>
      <td>3.35</td>
      <td>3.50</td>
      <td>11</td>
      <td>8</td>
      <td>1.375000</td>
      <td>154</td>
      <td>418</td>
      <td>0.368421</td>
      <td>-0.467527</td>
      <td>0.566745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Copper</td>
      <td>4.00</td>
      <td>97.648387</td>
      <td>73.00</td>
      <td>588.00</td>
      <td>52.00</td>
      <td>67.00</td>
      <td>8</td>
      <td>7</td>
      <td>1.142857</td>
      <td>158</td>
      <td>418</td>
      <td>0.377990</td>
      <td>2.303640</td>
      <td>7.624023</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alk_Phos</td>
      <td>289.00</td>
      <td>1982.655769</td>
      <td>1259.00</td>
      <td>13862.40</td>
      <td>601.00</td>
      <td>794.00</td>
      <td>2</td>
      <td>2</td>
      <td>1.000000</td>
      <td>295</td>
      <td>418</td>
      <td>0.705742</td>
      <td>2.992834</td>
      <td>9.662553</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SGOT</td>
      <td>26.35</td>
      <td>122.556346</td>
      <td>114.70</td>
      <td>457.25</td>
      <td>71.30</td>
      <td>137.95</td>
      <td>6</td>
      <td>5</td>
      <td>1.200000</td>
      <td>179</td>
      <td>418</td>
      <td>0.428230</td>
      <td>1.449197</td>
      <td>4.311976</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tryglicerides</td>
      <td>33.00</td>
      <td>124.702128</td>
      <td>108.00</td>
      <td>598.00</td>
      <td>118.00</td>
      <td>90.00</td>
      <td>7</td>
      <td>6</td>
      <td>1.166667</td>
      <td>146</td>
      <td>418</td>
      <td>0.349282</td>
      <td>2.523902</td>
      <td>11.802753</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Platelets</td>
      <td>62.00</td>
      <td>257.024570</td>
      <td>251.00</td>
      <td>721.00</td>
      <td>344.00</td>
      <td>269.00</td>
      <td>6</td>
      <td>5</td>
      <td>1.200000</td>
      <td>243</td>
      <td>418</td>
      <td>0.581340</td>
      <td>0.627098</td>
      <td>0.863045</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Prothrombin</td>
      <td>9.00</td>
      <td>10.731731</td>
      <td>10.60</td>
      <td>18.00</td>
      <td>10.60</td>
      <td>11.00</td>
      <td>39</td>
      <td>32</td>
      <td>1.218750</td>
      <td>48</td>
      <td>418</td>
      <td>0.114833</td>
      <td>2.223276</td>
      <td>10.040773</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stage</td>
      <td>1.00</td>
      <td>3.024272</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>155</td>
      <td>144</td>
      <td>1.076389</td>
      <td>4</td>
      <td>418</td>
      <td>0.009569</td>
      <td>-0.496273</td>
      <td>-0.638354</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cirrhosis_survival_object = cirrhosis_survival.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cirrhosis_survival_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cirrhosis_survival[x].value_counts().index.tolist()[0] for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cirrhosis_survival[x].value_counts().index.tolist()[1] for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cirrhosis_survival_object[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cirrhosis_survival_object[x].isin([cirrhosis_survival[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cirrhosis_survival_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cirrhosis_survival_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cirrhosis_survival_object)] * len(cirrhosis_survival_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Status</td>
      <td>C</td>
      <td>D</td>
      <td>232</td>
      <td>161</td>
      <td>1.440994</td>
      <td>3</td>
      <td>418</td>
      <td>0.007177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drug</td>
      <td>D-penicillamine</td>
      <td>Placebo</td>
      <td>158</td>
      <td>154</td>
      <td>1.025974</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sex</td>
      <td>F</td>
      <td>M</td>
      <td>374</td>
      <td>44</td>
      <td>8.500000</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascites</td>
      <td>N</td>
      <td>Y</td>
      <td>288</td>
      <td>24</td>
      <td>12.000000</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hepatomegaly</td>
      <td>Y</td>
      <td>N</td>
      <td>160</td>
      <td>152</td>
      <td>1.052632</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spiders</td>
      <td>N</td>
      <td>Y</td>
      <td>222</td>
      <td>90</td>
      <td>2.466667</td>
      <td>2</td>
      <td>418</td>
      <td>0.004785</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Edema</td>
      <td>N</td>
      <td>S</td>
      <td>354</td>
      <td>44</td>
      <td>8.045455</td>
      <td>3</td>
      <td>418</td>
      <td>0.007177</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    3




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



### 1.3.3. Data Preprocessing <a class="anchor" id="1.3.3"></a>

#### 1.3.3.1 Data Cleaning <a class="anchor" id="1.3.3.1"></a>


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival.shape)
```

    Dataset Dimensions: 
    


    (418, 19)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.40
##################################
cirrhosis_survival_filtered_row = cirrhosis_survival.drop(cirrhosis_survival[cirrhosis_survival.index.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cirrhosis_survival_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (312, 19)



```python
##################################
# Gathering the missing data percentage for each column
# from the filtered data
##################################
data_type_list = list(cirrhosis_survival_filtered_row.dtypes)
variable_name_list = list(cirrhosis_survival_filtered_row.columns)
null_count_list = list(cirrhosis_survival_filtered_row.isna().sum(axis=0))
non_null_count_list = list(cirrhosis_survival_filtered_row.count())
row_count_list = list([len(cirrhosis_survival_filtered_row)] * len(cirrhosis_survival_filtered_row.columns))
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary.sort_values(['Fill.Rate'], ascending=True))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>312</td>
      <td>282</td>
      <td>30</td>
      <td>0.903846</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>312</td>
      <td>284</td>
      <td>28</td>
      <td>0.910256</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>312</td>
      <td>308</td>
      <td>4</td>
      <td>0.987179</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>312</td>
      <td>310</td>
      <td>2</td>
      <td>0.993590</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cirrhosis_survival_cleaned = cirrhosis_survival_filtered_row
```

#### 1.3.3.2 Missing Data Imputation <a class="anchor" id="1.3.3.2"></a>

1. Missing data for float variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">Tryglicerides</span>: Null.Count = 30
    * <span style="color: #FF0000">Cholesterol</span>: Null.Count = 28
    * <span style="color: #FF0000">Platelets</span>: Null.Count = 4
    * <span style="color: #FF0000">Copper</span>: Null.Count = 2


```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_cleaned.columns),
                                                  list(cirrhosis_survival_cleaned.dtypes),
                                                  list([len(cirrhosis_survival_cleaned)] * len(cirrhosis_survival_cleaned.columns)),
                                                  list(cirrhosis_survival_cleaned.count()),
                                                  list(cirrhosis_survival_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary.sort_values(by=['Null.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>312</td>
      <td>282</td>
      <td>30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>312</td>
      <td>284</td>
      <td>28</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>312</td>
      <td>308</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>312</td>
      <td>310</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Status</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>int64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drug</td>
      <td>object</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>float64</td>
      <td>312</td>
      <td>312</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cirrhosis_survival_cleaned_object = cirrhosis_survival_cleaned.select_dtypes(include='object')
cirrhosis_survival_cleaned_object.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>Drug</th>
      <th>Sex</th>
      <th>Ascites</th>
      <th>Hepatomegaly</th>
      <th>Spiders</th>
      <th>Edema</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>M</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>D-penicillamine</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CL</td>
      <td>Placebo</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cirrhosis_survival_cleaned_int = cirrhosis_survival_cleaned.select_dtypes(include='int')
cirrhosis_survival_cleaned_int.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N_Days</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>400</td>
      <td>21464</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4500</td>
      <td>20617</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1012</td>
      <td>25594</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1925</td>
      <td>19994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1504</td>
      <td>13918</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cirrhosis_survival_cleaned_float = cirrhosis_survival_cleaned.select_dtypes(include='float')
cirrhosis_survival_cleaned_float.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
      <th>Prothrombin</th>
      <th>Stage</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14.5</td>
      <td>261.0</td>
      <td>2.60</td>
      <td>156.0</td>
      <td>1718.0</td>
      <td>137.95</td>
      <td>172.0</td>
      <td>190.0</td>
      <td>12.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.1</td>
      <td>302.0</td>
      <td>4.14</td>
      <td>54.0</td>
      <td>7394.8</td>
      <td>113.52</td>
      <td>88.0</td>
      <td>221.0</td>
      <td>10.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.4</td>
      <td>176.0</td>
      <td>3.48</td>
      <td>210.0</td>
      <td>516.0</td>
      <td>96.10</td>
      <td>55.0</td>
      <td>151.0</td>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.8</td>
      <td>244.0</td>
      <td>2.54</td>
      <td>64.0</td>
      <td>6121.8</td>
      <td>60.63</td>
      <td>92.0</td>
      <td>183.0</td>
      <td>10.3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.4</td>
      <td>279.0</td>
      <td>3.53</td>
      <td>143.0</td>
      <td>671.0</td>
      <td>113.15</td>
      <td>72.0</td>
      <td>136.0</td>
      <td>10.9</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cirrhosis_survival_imputed_float_array = iterative_imputer.fit_transform(cirrhosis_survival_cleaned_float)
```


```python
##################################
# Transforming the imputed data
# from an array to a dataframe
##################################
cirrhosis_survival_imputed_float = pd.DataFrame(cirrhosis_survival_imputed_float_array, 
                                                columns = cirrhosis_survival_cleaned_float.columns)
```


```python
##################################
# Formulating the imputed dataset
##################################
cirrhosis_survival_imputed = pd.concat([cirrhosis_survival_cleaned_int,
                                        cirrhosis_survival_cleaned_object,
                                        cirrhosis_survival_imputed_float], 
                                       axis=1, 
                                       join='inner')  
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
imputed_column_quality_summary = pd.DataFrame(zip(list(cirrhosis_survival_imputed.columns),
                                                  list(cirrhosis_survival_imputed.dtypes),
                                                  list([len(cirrhosis_survival_imputed)] * len(cirrhosis_survival_imputed.columns)),
                                                  list(cirrhosis_survival_imputed.count()),
                                                  list(cirrhosis_survival_imputed.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(imputed_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>int64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Status</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Drug</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ascites</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spiders</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Edema</td>
      <td>object</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Copper</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stage</td>
      <td>float64</td>
      <td>311</td>
      <td>311</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed for 4 numeric variables with Outlier.Ratio>0.05 and marginal to high Skewness.
    * <span style="color: #FF0000">Alk_Phos</span>: Outlier.Count = 35, Outlier.Ratio = 0.112, Skewness=+2.987
    * <span style="color: #FF0000">Bilirubin</span>: Outlier.Count = 30, Outlier.Ratio = 0.096, Skewness=+2.904
    * <span style="color: #FF0000">Cholesterol</span>: Outlier.Count = 22, Outlier.Ratio = 0.071, Skewness=+3.487
    * <span style="color: #FF0000">Copper</span>: Outlier.Count = 17, Outlier.Ratio = 0.055, Skewness=+2.319
2. Minimal number of outliers observed for 6 numeric variables with Outlier.Ratio>0.00 but <0.05 and normal to marginal Skewness.
    * <span style="color: #FF0000">Prothrombin</span>: Outlier.Count = 14, Outlier.Ratio = 0.045, Skewness=+1.749
    * <span style="color: #FF0000">Tryglicerides</span>: Outlier.Count = 13, Outlier.Ratio = 0.042, Skewness=+2.619
    * <span style="color: #FF0000">Albumin</span>: Outlier.Count = 11, Outlier.Ratio = 0.035, Skewness=-0.581
    * <span style="color: #FF0000">SGOT</span>: Outlier.Count = 7, Outlier.Ratio = 0.022, Skewness=+1.450
    * <span style="color: #FF0000">Platelets</span>: Outlier.Count = 4, Outlier.Ratio = 0.013, Skewness=+0.357
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 1, Outlier.Ratio = 0.003, Skewness=+0.167


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cirrhosis_survival_imputed_numeric = cirrhosis_survival_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = list(cirrhosis_survival_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cirrhosis_survival_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cirrhosis_survival_imputed_numeric_q1 = cirrhosis_survival_imputed_numeric.quantile(0.25)
cirrhosis_survival_imputed_numeric_q3 = cirrhosis_survival_imputed_numeric.quantile(0.75)
cirrhosis_survival_imputed_numeric_iqr = cirrhosis_survival_imputed_numeric_q3 - cirrhosis_survival_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((cirrhosis_survival_imputed_numeric < (cirrhosis_survival_imputed_numeric_q1 - 1.5 * cirrhosis_survival_imputed_numeric_iqr)) | (cirrhosis_survival_imputed_numeric > (cirrhosis_survival_imputed_numeric_q3 + 1.5 * cirrhosis_survival_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(cirrhosis_survival_imputed_numeric)] * len(cirrhosis_survival_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary.sort_values(by=['Outlier.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Alk_Phos</td>
      <td>2.987109</td>
      <td>35</td>
      <td>311</td>
      <td>0.112540</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bilirubin</td>
      <td>2.904006</td>
      <td>30</td>
      <td>311</td>
      <td>0.096463</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cholesterol</td>
      <td>3.487590</td>
      <td>22</td>
      <td>311</td>
      <td>0.070740</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Copper</td>
      <td>2.319390</td>
      <td>17</td>
      <td>311</td>
      <td>0.054662</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Prothrombin</td>
      <td>1.749358</td>
      <td>14</td>
      <td>311</td>
      <td>0.045016</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tryglicerides</td>
      <td>2.618650</td>
      <td>13</td>
      <td>311</td>
      <td>0.041801</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albumin</td>
      <td>-0.581905</td>
      <td>11</td>
      <td>311</td>
      <td>0.035370</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SGOT</td>
      <td>1.449979</td>
      <td>7</td>
      <td>311</td>
      <td>0.022508</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Platelets</td>
      <td>0.356677</td>
      <td>4</td>
      <td>311</td>
      <td>0.012862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>0.167578</td>
      <td>1</td>
      <td>311</td>
      <td>0.003215</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>0.372897</td>
      <td>0</td>
      <td>311</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stage</td>
      <td>-0.517547</td>
      <td>0</td>
      <td>311</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cirrhosis_survival_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_imputed_numeric, x=column)
```


    
![png](output_92_0.png)
    



    
![png](output_92_1.png)
    



    
![png](output_92_2.png)
    



    
![png](output_92_3.png)
    



    
![png](output_92_4.png)
    



    
![png](output_92_5.png)
    



    
![png](output_92_6.png)
    



    
![png](output_92_7.png)
    



    
![png](output_92_8.png)
    



    
![png](output_92_9.png)
    



    
![png](output_92_10.png)
    



    
![png](output_92_11.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

[Pearson’s Correlation Coefficient](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) is a parametric measure of the linear correlation for a pair of features by calculating the ratio between their covariance and the product of their standard deviations. The presence of high absolute correlation values indicate the univariate association between the numeric predictors and the numeric response.

1. All numeric variables were retained since majority reported sufficiently moderate and statistically significant correlation with no excessive multicollinearity.
2. Among pairwise combinations of numeric variables, the highest Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Copper</span>: Pearson.Correlation.Coefficient = +0.456
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">SGOT</span>: Pearson.Correlation.Coefficient = +0.444
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Tryglicerides</span>: Pearson.Correlation.Coefficient = +0.437
    * <span style="color: #FF0000">Birilubin</span> and <span style="color: #FF0000">Cholesterol</span>: Pearson.Correlation.Coefficient = +0.416


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'}) 
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cirrhosis_survival_imputed_numeric_correlation_pairs = {}
cirrhosis_survival_imputed_numeric_columns = cirrhosis_survival_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cirrhosis_survival_imputed_numeric_columns, 2):
    cirrhosis_survival_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cirrhosis_survival_imputed_numeric.loc[:, numeric_column_a], 
        cirrhosis_survival_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cirrhosis_survival_imputed_numeric_summary = cirrhosis_survival_imputed_numeric.from_dict(cirrhosis_survival_imputed_numeric_correlation_pairs, orient='index')
cirrhosis_survival_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cirrhosis_survival_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bilirubin_Copper</th>
      <td>0.456480</td>
      <td>2.057795e-17</td>
    </tr>
    <tr>
      <th>Bilirubin_SGOT</th>
      <td>0.444043</td>
      <td>1.852208e-16</td>
    </tr>
    <tr>
      <th>Bilirubin_Tryglicerides</th>
      <td>0.437218</td>
      <td>5.956896e-16</td>
    </tr>
    <tr>
      <th>Bilirubin_Cholesterol</th>
      <td>0.416075</td>
      <td>1.888065e-14</td>
    </tr>
    <tr>
      <th>Cholesterol_SGOT</th>
      <td>0.359592</td>
      <td>6.324822e-11</td>
    </tr>
    <tr>
      <th>Bilirubin_Prothrombin</th>
      <td>0.354754</td>
      <td>1.181651e-10</td>
    </tr>
    <tr>
      <th>Copper_SGOT</th>
      <td>0.293139</td>
      <td>1.403023e-07</td>
    </tr>
    <tr>
      <th>Cholesterol_Tryglicerides</th>
      <td>0.286235</td>
      <td>2.816557e-07</td>
    </tr>
    <tr>
      <th>Copper_Tryglicerides</th>
      <td>0.285441</td>
      <td>3.047992e-07</td>
    </tr>
    <tr>
      <th>Copper_Stage</th>
      <td>0.268252</td>
      <td>1.587568e-06</td>
    </tr>
    <tr>
      <th>Prothrombin_Stage</th>
      <td>0.257093</td>
      <td>4.367792e-06</td>
    </tr>
    <tr>
      <th>Bilirubin_Stage</th>
      <td>0.235346</td>
      <td>2.756487e-05</td>
    </tr>
    <tr>
      <th>Copper_Prothrombin</th>
      <td>0.216772</td>
      <td>1.165157e-04</td>
    </tr>
    <tr>
      <th>N_Days_Prothrombin</th>
      <td>0.207582</td>
      <td>2.275995e-04</td>
    </tr>
    <tr>
      <th>Albumin_Platelets</th>
      <td>0.204838</td>
      <td>2.764242e-04</td>
    </tr>
    <tr>
      <th>Copper_Alk_Phos</th>
      <td>0.188146</td>
      <td>8.546790e-04</td>
    </tr>
    <tr>
      <th>Cholesterol_Platelets</th>
      <td>0.186060</td>
      <td>9.778573e-04</td>
    </tr>
    <tr>
      <th>Alk_Phos_Tryglicerides</th>
      <td>0.184062</td>
      <td>1.110965e-03</td>
    </tr>
    <tr>
      <th>SGOT_Stage</th>
      <td>0.164321</td>
      <td>3.661498e-03</td>
    </tr>
    <tr>
      <th>Alk_Phos_Platelets</th>
      <td>0.146265</td>
      <td>9.796766e-03</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cirrhosis_survival_imputed_numeric_correlation = cirrhosis_survival_imputed_numeric.corr()
mask = np.triu(cirrhosis_survival_imputed_numeric_correlation)
plot_correlation_matrix(cirrhosis_survival_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_97_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cirrhosis_survival_imputed_numeric_correlation_p_values = correlation_significance(cirrhosis_survival_imputed_numeric)                     
mask = np.invert(np.tril(cirrhosis_survival_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cirrhosis_survival_imputed_numeric_correlation,mask)
```


    
![png](output_99_0.png)
    


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A Yeo-Johnson transformation was applied to all numeric variables to improve distributional shape.
2. Most variables achieved symmetrical distributions with minimal outliers after transformation.
    * <span style="color: #FF0000">Cholesterol</span>: Outlier.Count = 11, Outlier.Ratio = 0.035, Skewness=-0.059
    * <span style="color: #FF0000">Albumin</span>: Outlier.Count = 6, Outlier.Ratio = 0.019, Skewness=+0.020
    * <span style="color: #FF0000">Tryglicerides</span>: Outlier.Count = 6, Outlier.Ratio = 0.019, Skewness=-0.008
    * <span style="color: #FF0000">Copper</span>: Outlier.Count = 3, Outlier.Ratio = 0.010, Skewness=-0.001
    * <span style="color: #FF0000">SGOT</span>: Outlier.Count = 3, Outlier.Ratio = 0.010, Skewness=-0.001
    * <span style="color: #FF0000">Alk_Phos</span>: Outlier.Count = 2, Outlier.Ratio = 0.006, Skewness=+0.010
    * <span style="color: #FF0000">Platelets</span>: Outlier.Count = 1, Outlier.Ratio = 0.003, Skewness=+0.022
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 1, Outlier.Ratio = 0.003, Skewness=+0.167



```python
predictors_with_outliers = ['Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets']
cirrhosis_survival_imputed_numeric_with_outliers = cirrhosis_survival_imputed_numeric[predictors_with_outliers]
```


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cirrhosis_survival_imputed_numeric_with_outliers_array = yeo_johnson_transformer.fit_transform(cirrhosis_survival_imputed_numeric_with_outliers)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cirrhosis_survival_transformed_numeric_with_outliers = pd.DataFrame(cirrhosis_survival_imputed_numeric_with_outliers_array,
                                                                    columns=cirrhosis_survival_imputed_numeric_with_outliers.columns)
cirrhosis_survival_transformed_numeric = pd.concat([cirrhosis_survival_imputed_numeric[['N_Days','Age']],cirrhosis_survival_transformed_numeric_with_outliers], axis=1)
```


```python
cirrhosis_survival_transformed_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N_Days</th>
      <th>Age</th>
      <th>Bilirubin</th>
      <th>Cholesterol</th>
      <th>Albumin</th>
      <th>Copper</th>
      <th>Alk_Phos</th>
      <th>SGOT</th>
      <th>Tryglicerides</th>
      <th>Platelets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>400.0</td>
      <td>21464.0</td>
      <td>0.619732</td>
      <td>1.432144</td>
      <td>21.485967</td>
      <td>6.266174</td>
      <td>2.164598</td>
      <td>5.145246</td>
      <td>2.596994</td>
      <td>35.866346</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4500.0</td>
      <td>20617.0</td>
      <td>0.689000</td>
      <td>1.440905</td>
      <td>11.135260</td>
      <td>4.717665</td>
      <td>2.267788</td>
      <td>4.579478</td>
      <td>2.781609</td>
      <td>40.648181</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1012.0</td>
      <td>25594.0</td>
      <td>0.847264</td>
      <td>1.443980</td>
      <td>22.157066</td>
      <td>5.752213</td>
      <td>2.181419</td>
      <td>5.349723</td>
      <td>2.696238</td>
      <td>33.497223</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1925.0</td>
      <td>19994.0</td>
      <td>0.463815</td>
      <td>1.441293</td>
      <td>28.789864</td>
      <td>4.411479</td>
      <td>2.200631</td>
      <td>5.104434</td>
      <td>2.647759</td>
      <td>50.186573</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1504.0</td>
      <td>13918.0</td>
      <td>0.525336</td>
      <td>1.446977</td>
      <td>30.578643</td>
      <td>4.459750</td>
      <td>2.193314</td>
      <td>4.575880</td>
      <td>3.040744</td>
      <td>43.618671</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cirrhosis_survival_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cirrhosis_survival_transformed_numeric, x=column)
```


    
![png](output_105_0.png)
    



    
![png](output_105_1.png)
    



    
![png](output_105_2.png)
    



    
![png](output_105_3.png)
    



    
![png](output_105_4.png)
    



    
![png](output_105_5.png)
    



    
![png](output_105_6.png)
    



    
![png](output_105_7.png)
    



    
![png](output_105_8.png)
    



    
![png](output_105_9.png)
    



```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_variable_name_list = list(cirrhosis_survival_transformed_numeric.columns)
numeric_skewness_list = cirrhosis_survival_transformed_numeric.skew()
cirrhosis_survival_transformed_numeric_q1 = cirrhosis_survival_transformed_numeric.quantile(0.25)
cirrhosis_survival_transformed_numeric_q3 = cirrhosis_survival_transformed_numeric.quantile(0.75)
cirrhosis_survival_transformed_numeric_iqr = cirrhosis_survival_transformed_numeric_q3 - cirrhosis_survival_transformed_numeric_q1
numeric_outlier_count_list = ((cirrhosis_survival_transformed_numeric < (cirrhosis_survival_transformed_numeric_q1 - 1.5 * cirrhosis_survival_transformed_numeric_iqr)) | (cirrhosis_survival_transformed_numeric > (cirrhosis_survival_transformed_numeric_q3 + 1.5 * cirrhosis_survival_transformed_numeric_iqr))).sum()
numeric_row_count_list = list([len(cirrhosis_survival_transformed_numeric)] * len(cirrhosis_survival_transformed_numeric.columns))
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)

numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary.sort_values(by=['Outlier.Count'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Cholesterol</td>
      <td>-0.059158</td>
      <td>11</td>
      <td>312</td>
      <td>0.035256</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albumin</td>
      <td>0.020584</td>
      <td>6</td>
      <td>312</td>
      <td>0.019231</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tryglicerides</td>
      <td>-0.008452</td>
      <td>6</td>
      <td>312</td>
      <td>0.019231</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Copper</td>
      <td>-0.000884</td>
      <td>3</td>
      <td>312</td>
      <td>0.009615</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SGOT</td>
      <td>-0.000418</td>
      <td>3</td>
      <td>312</td>
      <td>0.009615</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alk_Phos</td>
      <td>0.010346</td>
      <td>2</td>
      <td>312</td>
      <td>0.006410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>0.167578</td>
      <td>1</td>
      <td>312</td>
      <td>0.003205</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Platelets</td>
      <td>-0.022121</td>
      <td>1</td>
      <td>312</td>
      <td>0.003205</td>
    </tr>
    <tr>
      <th>0</th>
      <td>N_Days</td>
      <td>0.372897</td>
      <td>0</td>
      <td>312</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bilirubin</td>
      <td>0.254036</td>
      <td>0</td>
      <td>312</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Clinical Prediction Models](http://clinicalpredictionmodels.org/) by Ewout Steyerberg
* **[Book]** [Survival Analysis: A Self-Learning Text](https://link.springer.com/book/10.1007/978-1-4419-6646-9/) by David Kleinbaum and Mitchel Klein
* **[Book]** [Applied Survival Analysis Using R](https://link.springer.com/book/10.1007/978-3-319-31245-3/) by Dirk Moore
* **[Python Library API]** [SciKit-Survival](https://pypi.org/project/scikit-survival/) by SciKit-Survival Team
* **[Article]** [Exploring Time-to-Event with Survival Analysis](https://towardsdatascience.com/exploring-time-to-event-with-survival-analysis-8b0a7a33a7be) by Olivia Tanuwidjaja (Towards Data Science)
* **[Article]** [The Complete Introduction to Survival Analysis in Python](https://towardsdatascience.com/the-complete-introduction-to-survival-analysis-in-python-7523e17737e6) by Marco Peixeiro (Towards Data Science)
* **[Article]** [Survival Analysis Simplified: Explaining and Applying with Python](https://medium.com/@zynp.atlii/survival-analysis-simplified-explaining-and-applying-with-python-7efacf86ba32) by Zeynep Atli (Towards Data Science)
* **[Article]** [Survival Analysis in Python (KM Estimate, Cox-PH and AFT Model)](https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d) by Rahul Raoniar (Medium)
* **[Article]** [How to Evaluate Survival Analysis Models)](https://towardsdatascience.com/how-to-evaluate-survival-analysis-models-dd67bc10caae) by Nicolo Cosimo Albanese (Towards Data Science)
* **[Article]** [Survival Analysis with Python Tutorial — How, What, When, and Why)](https://pub.towardsai.net/survival-analysis-with-python-tutorial-how-what-when-and-why-19a5cfb3c312) by Towards AI Team (Medium)
* **[Article]** [Survival Analysis: Predict Time-To-Event With Machine Learning)](https://towardsdatascience.com/survival-analysis-predict-time-to-event-with-machine-learning-part-i-ba52f9ab9a46) by Lina Faik (Medium)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 1](https://www.kdnuggets.com/2020/07/complete-guide-survival-analysis-python-part1.html) by Pratik Shukla (KDNuggets)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 2](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-2.html) by Pratik Shukla (KDNuggets)
* **[Article]** [A Complete Guide To Survival Analysis In Python, Part 3](https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html) by Pratik Shukla (KDNuggets)
* **[Kaggle Project]** [Survival Analysis with Cox Model Implementation](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Bryan Boulé (Kaggle)
* **[Kaggle Project]** [Survival Analysis](https://www.kaggle.com/code/gunesevitan/survival-analysis/notebook) by Gunes Evitan (Kaggle)
* **[Kaggle Project]** [Survival Analysis of Lung Cancer Patients](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Sayan Chakraborty (Kaggle)
* **[Kaggle Project]** [COVID-19 Cox Survival Regression](https://www.kaggle.com/code/bryanb/survival-analysis-with-cox-model-implementation/notebook) by Ilias Katsabalos (Kaggle)
* **[Kaggle Project]** [Liver Cirrhosis Prediction with XGboost & EDA](https://www.kaggle.com/code/arjunbhaybhang/liver-cirrhosis-prediction-with-xgboost-eda) by Arjun Bhaybang (Kaggle)
* **[Kaggle Project]** [Survival Models VS ML Models Benchmark - Churn Tel](https://www.kaggle.com/code/caralosal/survival-models-vs-ml-models-benchmark-churn-tel) by Carlos Alonso Salcedo (Kaggle)
* **[Publication]** [Survival Analysis Part I: Basic Concepts and First Analyses](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394262/) by Taane Clark (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part II: Multivariate Data Analysis – An Introduction to Concepts and Methods](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394368/) by Mike Bradburn (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part III: Multivariate Data Analysis – Choosing a Model and Assessing its Adequacy and Fit](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2376927/) by Mike Bradburn (British Journal of Cancer)
* **[Publication]** [Survival Analysis Part IV: Further Concepts and Methods in Survival Analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394469/) by Taane Clark (British Journal of Cancer)
* **[Course]** [Survival Analysis in Python](https://app.datacamp.com/learn/courses/survival-analysis-in-python) by Shae Wang (DataCamp)


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

