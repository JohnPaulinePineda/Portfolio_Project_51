***
# Supervised Learning : Modelling Right-Censored Survival Time and Status Responses for Prediction

***
### John Pauline Pineda <br> <br> *May 25, 2024*
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
