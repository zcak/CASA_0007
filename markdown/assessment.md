```python
############################
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sms
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt # For plotting
import numpy as np              # For working with numerical data
import sklearn.cluster as sklc  # For clustering
import sklearn.metrics as sklm  # For the silhouette score

from pygam import GAM, s, f
from pygam import LinearGAM
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#############################
data_filename = pd.read_csv('data2.csv')
df = pd.DataFrame(data_filename)

```


```python
# Let us check the distribution of the observations (COVID-19 case rate in local authority area i of England) we investigate
sns.displot(df, x = 'case_rate', kind="kde")
plt.savefig('case_rate')

# As the plot is heavily right-tailed (positively skewed), we apply a log-linear model in the analysis where the new dependent variable 
# is the natural logarithm of the observations of COVID-19 case rate in local authority area i of England

sns.displot(df, x = 'log_case_rate', kind="kde")
plt.savefig('log_case_rate')
# The second plot looks more similar to normal distirbution.
```


    
![png](output_1_0.png)
    



    
![png](output_1_1.png)
    



```python
# Check the outliers of the observations (COVID-19 case rate in local authority area i of England in month j)
case_rate= np.log(df['case_rate'])
Q1 =  case_rate.quantile(0.25)
Q3 = case_rate.quantile(0.75)
IQR = Q3 - Q1


case_rate_outliers = df['Lower Tier Local Authority'][(np.log(df['case_rate']) < Q1-1.5*IQR ) | (np.log(df['case_rate']) > Q3+1.5*IQR)]

case_rate_no_outliers = df['Lower Tier Local Authority'][(np.log(df['case_rate']) > Q1-1.5*IQR ) | (np.log(df['case_rate']) < Q3+1.5*IQR)]

case_rate_outliers
```




    Series([], Name: case_rate, dtype: float64)




```python
# And draw the boxplot to visualize the distribution of 'np.log(df['case_rate'])':
output_filename = 'boxplot'
figure_width, figure_height = 5,7
plt.figure(figsize=(figure_width,figure_height))
plt.xlim([0.75,1.25])
plt.xticks([])
plt.boxplot(df['case_rate'], manage_ticks=(False))
plt.savefig(output_filename)
```


    
![png](output_3_0.png)
    



```python
# Create dummy variables for time and region variables
Month = pd.get_dummies(df['month'])
Month = Month[['November','December']]

Region = pd.get_dummies(df['region'])
Region = Region[['East of England','North East', 'North West', 'South East', 'South West', 'East Midlands', 'West Midlands','Yorkshire and the Humber']]
# Create the variables for climate features: 'precipitation','humidity' and 'temp'
x_values = df[['precipitation','humidity','temperature']]

# Combine the variables for climate features with dummy variables for time and region variables
x_values = pd.concat([x_values, Month, Region], axis=1, sort=False)

# Create the dependent variable
y_values = np.log(df['case_rate'])
```


```python
# Let us firstly check the correlation between independent variables (e.g. create a heat map)
fig = plt.figure(figsize=(16,8))
sns.heatmap(x_values.corr(),annot = True,cmap='YlGnBu', linecolor='r',linewidth=0.5)
output_filename = 'correlation matrix'
plt.savefig(output_filename)
```


    
![png](output_5_0.png)
    



```python
# As the correlations between the dummy variable 'December' and the variable of temperature is larger than 0.07,
# we cannot assume that multicollinearity doesn't exist.
# Hence, we remove the dummy variable 'December' from the model.

# Create dummy variables for time and region variables and remove variables for 'December'
Month = pd.get_dummies(df['month'])
Month = Month['November']

Region = pd.get_dummies(df['region'])
Region = Region[['East of England','North East','North West','South East','South West','East Midlands','West Midlands','Yorkshire and the Humber']]

# df['humidity:temp'] = df['humidity']*df['temp']

# df['November:East Midlands'] = Month*Region['East Midlands']
x_values = df[['temperature', 'humidity','precipitation']]
x_values = pd.concat([x_values, Month, Region], axis=1, sort=False)

# Create the dependent variable
y_values = df['log_case_rate']

fig = plt.figure(figsize=(20,10))
sns.heatmap(x_values.corr(),annot = True,cmap='YlGnBu', linecolor='r',linewidth=0.5)

X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)

regression_model_b = regression_model_a.fit()

# Print a summary of the results:
print(regression_model_b.summary())
# As the p value of precipitation is larger than 0.05, we cannot reject the null hypothesis and remove precipitation from the model.
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_case_rate   R-squared:                       0.350
    Model:                            OLS   Adj. R-squared:                  0.293
    Method:                 Least Squares   F-statistic:                     6.141
    Date:                Mon, 18 Jan 2021   Prob (F-statistic):           1.47e-08
    Time:                        18:45:06   Log-Likelihood:                -89.488
    No. Observations:                 150   AIC:                             205.0
    Df Residuals:                     137   BIC:                             244.1
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ============================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------
    const                       11.4374      1.587      7.209      0.000       8.300      14.575
    temperature                 -0.1028      0.017     -6.134      0.000      -0.136      -0.070
    humidity                    -6.4021      1.841     -3.477      0.001     -10.043      -2.761
    precipitation                0.0063      0.002      2.560      0.012       0.001       0.011
    November                     0.3187      0.081      3.916      0.000       0.158       0.480
    East of England              0.1879      0.208      0.905      0.367      -0.223       0.598
    North East                   0.0920      0.241      0.382      0.703      -0.384       0.568
    North West                   0.1059      0.186      0.570      0.569      -0.261       0.473
    South East                   0.1772      0.201      0.883      0.379      -0.220       0.574
    South West                   0.0907      0.229      0.397      0.692      -0.362       0.543
    East Midlands                0.0460      0.212      0.218      0.828      -0.372       0.464
    West Midlands                0.1784      0.257      0.694      0.489      -0.330       0.687
    Yorkshire and the Humber     0.0412      0.231      0.179      0.859      -0.415       0.498
    ==============================================================================
    Omnibus:                        1.365   Durbin-Watson:                   2.193
    Prob(Omnibus):                  0.505   Jarque-Bera (JB):                1.426
    Skew:                           0.220   Prob(JB):                        0.490
    Kurtosis:                       2.815   Cond. No.                     3.32e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.32e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    


    
![png](output_6_1.png)
    



```python
# Check the model fitness by residuals vs fitted plot:
residuals = regression_model_b.resid
fitted = regression_model_b.fittedvalues
smoothed = lowess(residuals,fitted)
top3 = abs(residuals).sort_values(ascending = False)[:3]

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (8,7)
fig, ax = plt.subplots()
ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Residuals')
ax.set_xlabel('Fitted Values')
ax.set_title('Residuals vs. Fitted')
ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

for i in top3.index:
    ax.annotate(i,xy=(fitted[i],residuals[i]))

plt.show()

```


    
![png](output_7_0.png)
    

