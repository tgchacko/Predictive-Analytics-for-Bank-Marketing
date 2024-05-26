# Predictive-Analytics-for-Bank-Marketing

## Table of Contents

[Problem Statement](#problem-statement)

[Project Objectives](#project-objectives)

[Data Sources](#data-sources)

[Dataset Description Overview](#data-description-overview)

[Tools](#tools)

[EDA Steps](#eda-steps)

[Data Preprocessing Steps and Inspiration](#data-preprocessing-steps-and-inspiration)

[Graphs/Visualizations](#graphs-visualizations)

[Summary of Findings from Graphs](#summary-of-findings-from-graphs)

[Choosing the Algorithm for the Project](#choosing-the-algorithm-for-the-best-project)

[Assumptions](#assumptions)

[Model Evaluation Metrics](#model-evaluation-metrics)

[Results](#results)

[Recommendations to Improve Subscriptions](#recommendations-to-improve-subscriptions)

[Limitations](#limitations)

[Future Possibilities of the Project](#future-possibilities-of-the-project)

[References](#references)

## Problem Statement

The Portuguese bank is experiencing a revenue decline due to a decrease in client deposits. Term deposits are crucial as they allow banks to invest in higher-gain financial products and cross-sell other products. The goal is to identify clients with a higher likelihood of subscribing to term deposits and focus marketing efforts on them.

## Project Objectives

- Meet and Greet Data: Understanding the dataset.
- Prepare the Data: Preparing the data for analysis.
- Perform Exploratory Analysis: Visualizations and insights.
- Model the Data: Using machine learning algorithms.
- Validate and Implement Model: Ensuring model reliability.
- Optimize and Strategize: Improving model performance and marketing strategies.

## Data Sources

The dataset describes the results of Portugal bank marketing campaigns, primarily conducted via direct phone calls to offer term deposits. The target variable is whether the client agreed to place a deposit ('yes') or not ('no').

Source: [Bank Marketing Dataset](https://github.com/tgchacko/Predictive-Analytics-for-Bank-Marketing/blob/main/bank-additional-full.csv)

## Dataset Description Overview

- Instances(rows): 41188
- Features(columns): 21
- No Null Values

## Attributes

- Age: Age of the client (numeric)
- Job: Type of job (categorical)
- Marital: Marital status (categorical)
- Education: Educational qualification (categorical)
- Default: Whether the client has any unpaid credit (categorical)
- Housing: Whether the client has a housing loan (categorical)
- Loan: Whether the client has a personal loan (categorical)

### Current Campaign Data:

- Contact: Type of contact communication (categorical)
- Month: Last contact month of the year (categorical)
- Day_of_week: Last contact day of the week (categorical)
- Duration: Last contact duration in seconds (numeric) (Note: Duration highly affects the output target but is only known after the call, thus should be discarded for realistic predictive models)

### Other Attributes:

- Campaign: Number of contacts performed during this campaign for this client (numeric)
- Pdays: Number of days since the client was last contacted from a previous campaign (numeric; 999 means the client was not previously contacted)
- Previous: Number of contacts performed before this campaign for this client (numeric)
- Poutcome: Outcome of the previous marketing campaign (categorical)

### Social and Economic Context:

- Emp.var.rate: Employment variation rate - quarterly indicator (numeric)
- Cons.price.idx: Consumer price index - monthly indicator (numeric)
- conf.idx: Consumer confidence index - monthly indicator (numeric)
- Euribor3m: Euribor 3 month rate - daily indicator (numeric)
- employed: Number of employees - quarterly indicator (numeric)

### Output Variable:

- y: Has the client subscribed to a term deposit? (binary: 'yes', 'no')

![Data Description](https://i.postimg.cc/vZx33gPh/Screenshot-2024-05-26-at-13-00-53-Bank-Jupyter-Notebook.png)

### Tools

- Python: Data Cleaning and Analysis

    [Download Python](https://www.python.org/downloads/)

- Jupyter Notebook: For interactive data analysis and visualization

    [Install Jupyter](https://jupyter.org/install)
 
**Libraries**

Below are the links for details and commands (if required) to install the necessary Python packages:

Below are the links for details and commands (if required) to install the necessary Python packages:
- **pandas**: Go to [Pandas Installation](https://pypi.org/project/pandas/) or use command: `pip install pandas`
- **numpy**: Go to [NumPy Installation](https://pypi.org/project/numpy/) or use command: `pip install numpy`
- **matplotlib**: Go to [Matplotlib Installation](https://pypi.org/project/matplotlib/) or use command: `pip install matplotlib`
- **seaborn**: Go to [Seaborn Installation](https://pypi.org/project/seaborn/) or use command: `pip install seaborn`
- **scikit-learn**: Go to [Scikit-Learn Installation](https://pypi.org/project/scikit-learn/) or use command: `pip install scikit-learn`
- **XGBoost**: Go to [XGBoost Installation](https://pypi.org/project/xgboost/) or use command: pip install xgboost
-	**Imbalanced-learn**: Go to [Imbalanced-learn Installation](https://pypi.org/project/imbalanced-learn/) or use command: pip install imbalanced-learn

## EDA Steps

- Data loading and initial exploration
- Data cleaning and manipulation
- Data visualization to understand feature distributions and relationships
- Identifying and handling missing and duplcicate values
- Checking for data imbalances

## Data Preprocessing Steps and Inspiration

-	**Handling Missing/ Duplicate Values**: Handling any missing/duplicate values in the dataset.
-	**Encoding Categorical Variables**: Label encoding to convert categorical variables into numeric format.
-	**SMOTE (Synthetic Minority Over-sampling Technique)**: It is used to address class imbalance by generating synthetic samples for the minority class, thus balancing the class distribution
-	**Scaling Numerical Features**: StandardScaler to standardize numerical features.
-	**Splitting the Dataset**: Splitting the dataset into training and testing sets using train_test_split.

## Graphs/Visualizations

![Ratio between different job types](https://i.postimg.cc/pTdm28nM/Screenshot-2024-05-26-at-13-10-55-Bank-Jupyter-Notebook.png)

![Distribution of Marital Statuses](https://i.postimg.cc/3rcLmQ72/Screenshot-2024-05-26-at-13-13-38-Bank-Jupyter-Notebook.png)

![Distribution of Education Levels among Clients](https://i.postimg.cc/wBLcw0PT/Screenshot-2024-05-26-at-18-28-11-Bank-Jupyter-Notebook.png)

![Percentage of Clients with Credit in Default](https://i.postimg.cc/CMGMwkyB/Screenshot-2024-05-26-at-18-29-37-Bank-Jupyter-Notebook.png)

![Distribution of Housing Loan](https://i.postimg.cc/nzsTfB3C/Screenshot-2024-05-26-at-18-31-53-Bank-Jupyter-Notebook.png)

![Distribution of Personal Loan](https://i.postimg.cc/9Mt0MN8S/Screenshot-2024-05-26-at-18-32-27-Bank-Jupyter-Notebook.png)

![Number of Clients Contacted in Each Month](https://i.postimg.cc/L6f96WJV/Screenshot-2024-05-26-at-18-33-07-Bank-Jupyter-Notebook.png)

![Distribution of Last Contact Day of the Week](https://i.postimg.cc/RVm8KpPd/Screenshot-2024-05-26-at-18-34-07-Bank-Jupyter-Notebook.png)

![Ratio of clients who subscribed to a term deposit compared to those who did not](https://i.postimg.cc/7Yp5K72K/Screenshot-2024-05-26-at-18-35-04-Bank-Jupyter-Notebook.png)

![Distribution of Ages Among Clients](https://i.postimg.cc/vZ0HpVgR/Screenshot-2024-05-26-at-18-35-56-Bank-Jupyter-Notebook.png)

![Subscription Counts by Job Type](https://i.postimg.cc/VLbqsz9H/Screenshot-2024-05-26-at-18-37-16-Bank-Jupyter-Notebook.png)

![Subscription Counts by Education](https://i.postimg.cc/tg1Zk5rv/Screenshot-2024-05-26-at-18-38-28-Bank-Jupyter-Notebook.png)

![Subscription Counts by Marital Status](https://i.postimg.cc/Y2PPyqqX/Screenshot-2024-05-26-at-18-39-58-Bank-Jupyter-Notebook.png)

![Correlation Matrix Heatmap](https://i.postimg.cc/VkXLkVV8/Untitled456.png)

![Boxplots](https://i.postimg.cc/PfmCxNpc/Screenshot-2024-05-26-at-18-42-47-Bank-Jupyter-Notebook.png)

![Job vs Duration](https://i.postimg.cc/wMdcmPkd/Screenshot-2024-05-26-at-18-43-46-Bank-Jupyter-Notebook.png)

![Campaign vs Duration](https://i.postimg.cc/FHQ5KsSP/Screenshot-2024-05-26-at-18-44-38-Bank-Jupyter-Notebook.png)

![Campaign vs Month](https://i.postimg.cc/CLnngJCG/Screenshot-2024-05-26-at-18-45-12-Bank-Jupyter-Notebook.png)

## Summary of Findings from Graphs

### Campaign Timing:
- The campaign operated only on weekdays.
- Most campaign activities were concentrated at the beginning of the bank period (May, June, and July), potentially aligning with parents making deposits for their children's education.

### Call Duration and Deposits:
- Leads who did not make deposits had shorter call durations.
- Longer call durations were associated with a higher probability of making a deposit.
- Average call durations were highest for blue-collar workers and entrepreneurs, and lowest for students and retirees.

### Lead Distribution:
- A large number of leads came from self-employed clients and management personnel.
- Many positive leads were observed in the initial days of the campaign.

### Economic Indicators:
- The campaign was conducted during a period of high employee variation rate, indicating job shifts due to economic conditions.
- The consumer price index was favorable, suggesting that leads had the financial capability to make deposits.
- The consumer confidence index was low, reflecting economic uncertainty.
- The 3-month Euribor interest rate was high, affecting loan interest rates.
- The number of employees was at its peak, possibly increasing income levels and targeting employed leads for deposits.

### Marital Status and Deposits:
- Married leads had a higher tendency to make deposits, followed by single leads.
- Married leads also showed a higher price index contribution as a couple.

### Job and Education:
- Deposits were primarily made by leads in administrative positions, followed by technicians and blue-collar workers.
- Leads with at least a university degree were more likely to make deposits, followed by those with a high school education.

### Campaign Effectiveness:
- There were more deposits made in May, the start of the bank period.
- The campaign effectively targeted leads with stable economic conditions and higher education levels, leading to successful deposit conversions.

## Choosing the Algorithm for the Project

- **Logistic Regression**: A statistical model that estimates the probability of a binary outcome based on one or more predictor variables.
- **Decision Tree Classifier**: A tree-like model used to make decisions based on the features of the data.
- **Random Forest Classifier**: An ensemble method that creates a forest of decision trees and outputs the mode of their predictions.
- **Neighbors Classifier (KNN)**: A simple, instance-based learning algorithm that assigns class labels based on the majority vote of its neighbors.
- **Naive Bayes Classifier**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions.
- **XGBoost Classifier**: An optimized gradient boosting library designed to be highly efficient and flexible.

## Assumptions

- The dataset provided is representative of the customer base.
- All relevant features are included in the dataset.
- The preprocessing steps adequately prepare the data for modeling.

## Model Evaluation Metrics

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The weighted average of Precision and Recall.
- **ROC-AUC Score**: A performance measurement for classification problems at various threshold settings.

## Results

### Without Applying SMOTE

![Results without SMOTE Smote](https://i.postimg.cc/G2R6cMsH/Screenshot-2024-05-26-at-18-46-29-Bank-Jupyter-Notebook.png)

### After Applying SMOTE

![Results with Smote](https://i.postimg.cc/Y0HqxQ9J/Screenshot-2024-05-26-at-18-49-10-Bank-Jupyter-Notebook.png)

The models' performance was evaluated using the above metrics was the Random Forest Classifier(after applying SMOTE) with the following findings:
- **Accuracy**: 97.52%
- **Precision**: 96.43%
- **Recall**: 98.74%
- **ROC-AUC Score**: 97.51%

## Recommendations to Improve Subscriptions

### Targeted Job Roles:
- Classify job roles based on corporate tiers.
- Approach tier 1 employees within a few days after the campaign commences.

### Enhance Call Engagement:
- Listen actively to leads to gather more information.
- Deliver tailored deposit plans to increase call duration and conversion rates.

### Optimal Campaign Timing:
- Focus campaign efforts during the start of the new bank period (May-July) when data shows positive results.

### Economic Considerations:
- Adjust campaign strategies according to national economic conditions.
- Avoid heavy campaign spending during poor economic performance.

### Continuous Improvement:
- Further data collection and feature engineering to enhance model performance.
- Regularly update the model with new data to maintain accuracy.
- Implement retention strategies based on model predictions to reduce customer attrition.

## Limitations

- The dataset may contain biases that could affect the model's predictions.
- The models' performance is limited by the quality and quantity of the available data.

## Future Possibilities of the Project

- Exploring additional algorithms and ensemble methods
- Implementing deep learning models for better performance
- Automating the model updating process with new incoming data
- Developing real-time customer behavior prediction systems

## References

- [Scikit-learn documentation](https://pypi.org/project/imbalanced-learn/)
- [XGBoost documentation](https://pypi.org/project/xgboost/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)




