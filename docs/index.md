***
# Supervised Learning : Modelling Right-Censored Survival Time and Status Responses for Prediction

***
### John Pauline Pineda <br> <br> *June 8, 2024*
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

## 1.2. Data Description <a class="anchor" id="1.2"></a>


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
      <th>ID</th>
      <td>418.0</td>
      <td>209.500000</td>
      <td>120.810458</td>
      <td>1.00</td>
      <td>105.2500</td>
      <td>209.50</td>
      <td>313.75</td>
      <td>418.00</td>
    </tr>
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
      <td>ID</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N_Days</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Status</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Age</td>
      <td>int64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sex</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Edema</td>
      <td>object</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bilirubin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Albumin</td>
      <td>float64</td>
      <td>418</td>
      <td>418</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prothrombin</td>
      <td>float64</td>
      <td>418</td>
      <td>416</td>
      <td>2</td>
      <td>0.995215</td>
    </tr>
    <tr>
      <th>19</th>
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
print('Number of Columns with Missing Data:',str(len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])))
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
      <th>16</th>
      <td>Tryglicerides</td>
      <td>float64</td>
      <td>418</td>
      <td>282</td>
      <td>136</td>
      <td>0.674641</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cholesterol</td>
      <td>float64</td>
      <td>418</td>
      <td>284</td>
      <td>134</td>
      <td>0.679426</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Copper</td>
      <td>float64</td>
      <td>418</td>
      <td>310</td>
      <td>108</td>
      <td>0.741627</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Drug</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ascites</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hepatomegaly</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Spiders</td>
      <td>object</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Alk_Phos</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SGOT</td>
      <td>float64</td>
      <td>418</td>
      <td>312</td>
      <td>106</td>
      <td>0.746411</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Platelets</td>
      <td>float64</td>
      <td>418</td>
      <td>407</td>
      <td>11</td>
      <td>0.973684</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Stage</td>
      <td>float64</td>
      <td>418</td>
      <td>412</td>
      <td>6</td>
      <td>0.985646</td>
    </tr>
    <tr>
      <th>18</th>
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
row_metadata_list = cirrhosis_survival["ID"].values.tolist()
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
      <td>20</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>20</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>0.00</td>
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
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
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
      <td>20</td>
      <td>1</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>20</td>
      <td>2</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>20</td>
      <td>2</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>20</td>
      <td>2</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>20</td>
      <td>2</td>
      <td>0.10</td>
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
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>414</th>
      <td>415</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>415</th>
      <td>416</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>416</th>
      <td>417</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>417</th>
      <td>418</td>
      <td>20</td>
      <td>9</td>
      <td>0.45</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 4 columns</p>
</div>



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
      <td>ID</td>
      <td>1.00</td>
      <td>209.500000</td>
      <td>209.50</td>
      <td>418.00</td>
      <td>1.00</td>
      <td>314.00</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>418</td>
      <td>418</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>-1.200000</td>
    </tr>
    <tr>
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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
      <th>6</th>
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
      <th>7</th>
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
      <th>8</th>
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
      <th>9</th>
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
      <th>10</th>
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
      <th>11</th>
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
      <th>12</th>
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

