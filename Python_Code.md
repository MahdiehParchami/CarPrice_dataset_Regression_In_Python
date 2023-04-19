```python
# import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd #data processing


from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
from scipy import stats


#Visualization and plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
```


```python
# loading the data

df_car = pd.read_csv('D:/Mahdieh_CourseUniversity/University_courses/ALY6020/Module_2/MidWeek/CarPrice_Assignment.csv')
df_car
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CarName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>alfa-romero giulia</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>alfa-romero stelvio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>alfa-romero Quadrifoglio</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>audi 100 ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>audi 100ls</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>201</td>
      <td>-1</td>
      <td>volvo 145e (sw)</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845.0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>202</td>
      <td>-1</td>
      <td>volvo 144ea</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045.0</td>
    </tr>
    <tr>
      <th>202</th>
      <td>203</td>
      <td>-1</td>
      <td>volvo 244dl</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>204</td>
      <td>-1</td>
      <td>volvo 246</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>205</td>
      <td>-1</td>
      <td>volvo 264gl</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625.0</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 26 columns</p>
</div>




```python
#descriptive analysis 

df_car.describe()
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>wheelbase</th>
      <th>carlength</th>
      <th>carwidth</th>
      <th>carheight</th>
      <th>curbweight</th>
      <th>enginesize</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>103.000000</td>
      <td>0.834146</td>
      <td>98.756585</td>
      <td>174.049268</td>
      <td>65.907805</td>
      <td>53.724878</td>
      <td>2555.565854</td>
      <td>126.907317</td>
      <td>3.329756</td>
      <td>3.255415</td>
      <td>10.142537</td>
      <td>104.117073</td>
      <td>5125.121951</td>
      <td>25.219512</td>
      <td>30.751220</td>
      <td>13276.710571</td>
    </tr>
    <tr>
      <th>std</th>
      <td>59.322565</td>
      <td>1.245307</td>
      <td>6.021776</td>
      <td>12.337289</td>
      <td>2.145204</td>
      <td>2.443522</td>
      <td>520.680204</td>
      <td>41.642693</td>
      <td>0.270844</td>
      <td>0.313597</td>
      <td>3.972040</td>
      <td>39.544167</td>
      <td>476.985643</td>
      <td>6.542142</td>
      <td>6.886443</td>
      <td>7988.852332</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>2.540000</td>
      <td>2.070000</td>
      <td>7.000000</td>
      <td>48.000000</td>
      <td>4150.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>5118.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.100000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>3.150000</td>
      <td>3.110000</td>
      <td>8.600000</td>
      <td>70.000000</td>
      <td>4800.000000</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>7788.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>103.000000</td>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>120.000000</td>
      <td>3.310000</td>
      <td>3.290000</td>
      <td>9.000000</td>
      <td>95.000000</td>
      <td>5200.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>10295.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>154.000000</td>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.100000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2935.000000</td>
      <td>141.000000</td>
      <td>3.580000</td>
      <td>3.410000</td>
      <td>9.400000</td>
      <td>116.000000</td>
      <td>5500.000000</td>
      <td>30.000000</td>
      <td>34.000000</td>
      <td>16503.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>205.000000</td>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>3.940000</td>
      <td>4.170000</td>
      <td>23.000000</td>
      <td>288.000000</td>
      <td>6600.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>45400.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Check type of variables and find null values

df_car.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   car_ID            205 non-null    int64  
     1   symboling         205 non-null    int64  
     2   CarName           205 non-null    object 
     3   fueltype          205 non-null    object 
     4   aspiration        205 non-null    object 
     5   doornumber        205 non-null    object 
     6   carbody           205 non-null    object 
     7   drivewheel        205 non-null    object 
     8   enginelocation    205 non-null    object 
     9   wheelbase         205 non-null    float64
     10  carlength         205 non-null    float64
     11  carwidth          205 non-null    float64
     12  carheight         205 non-null    float64
     13  curbweight        205 non-null    int64  
     14  enginetype        205 non-null    object 
     15  cylindernumber    205 non-null    object 
     16  enginesize        205 non-null    int64  
     17  fuelsystem        205 non-null    object 
     18  boreratio         205 non-null    float64
     19  stroke            205 non-null    float64
     20  compressionratio  205 non-null    float64
     21  horsepower        205 non-null    int64  
     22  peakrpm           205 non-null    int64  
     23  citympg           205 non-null    int64  
     24  highwaympg        205 non-null    int64  
     25  price             205 non-null    float64
    dtypes: float64(8), int64(8), object(10)
    memory usage: 41.8+ KB
    


```python
df_car.shape
```




    (205, 26)




```python
#Check missing values

df_car.isnull().values.any()
```




    False




```python
#Find the missing values

df_car.isnull().sum()
```




    car_ID              0
    symboling           0
    CarName             0
    fueltype            0
    aspiration          0
    doornumber          0
    carbody             0
    drivewheel          0
    enginelocation      0
    wheelbase           0
    carlength           0
    carwidth            0
    carheight           0
    curbweight          0
    enginetype          0
    cylindernumber      0
    enginesize          0
    fuelsystem          0
    boreratio           0
    stroke              0
    compressionratio    0
    horsepower          0
    peakrpm             0
    citympg             0
    highwaympg          0
    price               0
    dtype: int64




```python
#Visualize missing values

plt.figure(figsize=(6,4))  #create a new figure to see the plot based on this size
sns.heatmap(df_car.isnull())  # see the rows of missing values in a heatmap
```




    <AxesSubplot: >




    
![png](output_7_1.png)
    



```python
#Checking for duplicates

df_car.loc[df_car.duplicated()]
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CompanyName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 26 columns</p>
</div>




```python
#Splitting company name from CarName column

CompanyName = df_car['CarName'].apply(lambda x : x.split(' ')[0])
df_car.insert(3,"CompanyName",CompanyName)
df_car.drop(['CarName'],axis=1,inplace=True)
df_car.head()
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
      <th>car_ID</th>
      <th>symboling</th>
      <th>CompanyName</th>
      <th>fueltype</th>
      <th>aspiration</th>
      <th>doornumber</th>
      <th>carbody</th>
      <th>drivewheel</th>
      <th>enginelocation</th>
      <th>wheelbase</th>
      <th>...</th>
      <th>enginesize</th>
      <th>fuelsystem</th>
      <th>boreratio</th>
      <th>stroke</th>
      <th>compressionratio</th>
      <th>horsepower</th>
      <th>peakrpm</th>
      <th>citympg</th>
      <th>highwaympg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
df_car['CompanyName'].unique()
```




    array(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
           'isuzu', 'jaguar', 'maxda', 'mazda', 'buick', 'mercury',
           'mitsubishi', 'Nissan', 'nissan', 'peugeot', 'plymouth', 'porsche',
           'porcshce', 'renault', 'saab', 'subaru', 'toyota', 'toyouta',
           'vokswagen', 'volkswagen', 'vw', 'volvo'], dtype=object)




```python
# Fixing invalid values¶
#There seems to be some spelling error in the CompanyName column.

#maxda = mazda
#Nissan = nissan
#porsche = porcshce
#toyota = toyouta
#vokswagen = volkswagen = vw

df_car.CompanyName = df_car.CompanyName.str.lower()

def replace_name(a,b):
    df_car.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

df_car.CompanyName.unique()
```




    array(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
           'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi',
           'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab',
           'subaru', 'toyota', 'volkswagen', 'volvo'], dtype=object)




```python
# Separating categorical Variables

cat_variables = [x for x in df_car.columns if df_car[x].dtype == "object"]
cat_variables
```




    ['CompanyName',
     'fueltype',
     'aspiration',
     'doornumber',
     'carbody',
     'drivewheel',
     'enginelocation',
     'enginetype',
     'cylindernumber',
     'fuelsystem']




```python
# Separating Numeric Variables

num_variables = [x for x in df_car.columns if x not in cat_variables]
num_variables
```




    ['car_ID',
     'symboling',
     'wheelbase',
     'carlength',
     'carwidth',
     'carheight',
     'curbweight',
     'enginesize',
     'boreratio',
     'stroke',
     'compressionratio',
     'horsepower',
     'peakrpm',
     'citympg',
     'highwaympg',
     'price']




```python
#Checking the unique values and their counts in each categorical variable

for i in cat_variables:
    print("------"+i+"---------")
    print(df_car[i].value_counts(normalize=True))
    print("\n")
```

    ------CompanyName---------
    toyota         0.156098
    nissan         0.087805
    mazda          0.082927
    mitsubishi     0.063415
    honda          0.063415
    volkswagen     0.058537
    subaru         0.058537
    peugeot        0.053659
    volvo          0.053659
    dodge          0.043902
    buick          0.039024
    bmw            0.039024
    audi           0.034146
    plymouth       0.034146
    saab           0.029268
    porsche        0.024390
    isuzu          0.019512
    jaguar         0.014634
    chevrolet      0.014634
    alfa-romero    0.014634
    renault        0.009756
    mercury        0.004878
    Name: CompanyName, dtype: float64
    
    
    ------fueltype---------
    gas       0.902439
    diesel    0.097561
    Name: fueltype, dtype: float64
    
    
    ------aspiration---------
    std      0.819512
    turbo    0.180488
    Name: aspiration, dtype: float64
    
    
    ------doornumber---------
    four    0.560976
    two     0.439024
    Name: doornumber, dtype: float64
    
    
    ------carbody---------
    sedan          0.468293
    hatchback      0.341463
    wagon          0.121951
    hardtop        0.039024
    convertible    0.029268
    Name: carbody, dtype: float64
    
    
    ------drivewheel---------
    fwd    0.585366
    rwd    0.370732
    4wd    0.043902
    Name: drivewheel, dtype: float64
    
    
    ------enginelocation---------
    front    0.985366
    rear     0.014634
    Name: enginelocation, dtype: float64
    
    
    ------enginetype---------
    ohc      0.721951
    ohcf     0.073171
    ohcv     0.063415
    dohc     0.058537
    l        0.058537
    rotor    0.019512
    dohcv    0.004878
    Name: enginetype, dtype: float64
    
    
    ------cylindernumber---------
    four      0.775610
    six       0.117073
    five      0.053659
    eight     0.024390
    two       0.019512
    three     0.004878
    twelve    0.004878
    Name: cylindernumber, dtype: float64
    
    
    ------fuelsystem---------
    mpfi    0.458537
    2bbl    0.321951
    idi     0.097561
    1bbl    0.053659
    spdi    0.043902
    4bbl    0.014634
    mfi     0.004878
    spfi    0.004878
    Name: fuelsystem, dtype: float64
    
    
    


```python
#distribution of numerical variables

df_car.hist(edgecolor='black', linewidth=0.75)
fig=plt.gcf()
fig.set_size_inches(13,13)
plt.show()
```


    
![png](output_15_0.png)
    



```python
#check for outliers with selected numerical variables 


plt.figure(figsize=(8,6))
df_car.boxplot(column =['price','carlength','carwidth','carheight'],grid = False)
```




    <AxesSubplot: >




    
![png](output_16_1.png)
    



```python
# visualize categorical variables with their count

for i in cat_variables:
    plt.figure(figsize=(10,13))
    plt.title(i)
    sns.countplot(x = df_car[i] , order = df_car[i].value_counts().index )
    
    plt.xticks(rotation = 90, fontsize = 7)
    plt.show()
```


    
![png](output_17_0.png)
    



    
![png](output_17_1.png)
    



    
![png](output_17_2.png)
    



    
![png](output_17_3.png)
    



    
![png](output_17_4.png)
    



    
![png](output_17_5.png)
    



    
![png](output_17_6.png)
    



    
![png](output_17_7.png)
    



    
![png](output_17_8.png)
    



    
![png](output_17_9.png)
    



```python
# visualize categorical variables VS price

for i in cat_variables:
    plt.figure(figsize=(13,13))
    plt.title(i)
    sns.boxplot(x = df_car[i] , y= df_car['price'], order = df_car[i].value_counts().index, palette=("cubehelix") )
    
    plt.xticks(rotation = 90, fontsize = 10)
    plt.show()
```


    
![png](output_18_0.png)
    



    
![png](output_18_1.png)
    



    
![png](output_18_2.png)
    



    
![png](output_18_3.png)
    



    
![png](output_18_4.png)
    



    
![png](output_18_5.png)
    



    
![png](output_18_6.png)
    



    
![png](output_18_7.png)
    



    
![png](output_18_8.png)
    



    
![png](output_18_9.png)
    



```python
# visualize pairplot numerical variables VS price

def pp(x,y,z):
    sns.pairplot(df_car, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter', hue = 'fueltype')
    plt.show()

pp('carlength', 'carwidth', 'carheight')
pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    



    
![png](output_19_2.png)
    



    
![png](output_19_3.png)
    



```python
#check correlation

plt.figure(figsize=(12,12))
sns.heatmap(df_car.corr(),annot=True,cmap='Blues')
plt.show()
```


    
![png](output_20_0.png)
    



```python
#replace character with number for doornumber , cylindernumber

df_car['doornumber'].replace({'two':2, 'four':4}, inplace=True)
df_car['cylindernumber'].replace({'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}, inplace=True)
```

Perform Linear Regression Model


```python
#convert categorical to dummy variable 

df_car = pd.get_dummies(df_car, columns = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem','CompanyName'])
```


```python
y = df_car['price']
x = df_car.drop(['price'] , axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

```


```python
#add constant to predictor variables
x2 = sm.add_constant(x_train)

#fit regression model

model = sm.OLS(y_train,x2).fit()

#view summary of model fit

print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.972
    Model:                            OLS   Adj. R-squared:                  0.955
    Method:                 Least Squares   F-statistic:                     56.20
    Date:                Sun, 22 Jan 2023   Prob (F-statistic):           2.49e-57
    Time:                        16:53:24   Log-Likelihood:                -1412.4
    No. Observations:                 164   AIC:                             2951.
    Df Residuals:                     101   BIC:                             3146.
    Df Model:                          62                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    const                   -9328.6080   5225.663     -1.785      0.077   -1.97e+04    1037.702
    car_ID                    110.8593     60.822      1.823      0.071      -9.795     231.514
    symboling                 267.0289    303.842      0.879      0.382    -335.712     869.770
    doornumber                495.7396    275.961      1.796      0.075     -51.692    1043.172
    wheelbase                 263.0326    105.109      2.502      0.014      54.525     471.540
    carlength                -137.6630     52.728     -2.611      0.010    -242.261     -33.065
    carwidth                  452.9721    270.826      1.673      0.098     -84.274     990.218
    carheight                   3.0428    178.709      0.017      0.986    -351.468     357.553
    curbweight                  3.2967      1.746      1.888      0.062      -0.168       6.761
    cylindernumber            270.0655    762.544      0.354      0.724   -1242.616    1782.747
    enginesize                 84.0781     25.696      3.272      0.001      33.104     135.052
    boreratio               -1808.4413   1557.185     -1.161      0.248   -4897.478    1280.596
    stroke                  -1636.6814   1080.022     -1.515      0.133   -3779.155     505.793
    compressionratio        -1091.3345    476.963     -2.288      0.024   -2037.501    -145.168
    horsepower                  6.5830     25.309      0.260      0.795     -43.623      56.789
    peakrpm                     1.7307      0.724      2.391      0.019       0.295       3.167
    citympg                    85.8639    147.052      0.584      0.561    -205.848     377.575
    highwaympg                 60.1404    130.053      0.462      0.645    -197.849     318.130
    fueltype_diesel          1201.1596   2927.961      0.410      0.683   -4607.127    7009.447
    fueltype_gas            -1.053e+04   3819.775     -2.757      0.007   -1.81e+04   -2952.362
    aspiration_std          -5223.9810   2735.226     -1.910      0.059   -1.06e+04     201.971
    aspiration_turbo        -4104.6270   2578.882     -1.592      0.115   -9220.435    1011.181
    carbody_convertible       479.9061   1368.313      0.351      0.727   -2234.459    3194.272
    carbody_hardtop         -1674.6800   1390.497     -1.204      0.231   -4433.052    1083.692
    carbody_hatchback       -2933.6403   1090.884     -2.689      0.008   -5097.660    -769.621
    carbody_sedan           -2571.4338   1199.266     -2.144      0.034   -4950.455    -192.413
    carbody_wagon           -2628.7599   1353.609     -1.942      0.055   -5313.955      56.435
    drivewheel_4wd          -3261.2302   1839.035     -1.773      0.079   -6909.381     386.921
    drivewheel_fwd          -3293.2653   1896.598     -1.736      0.086   -7055.606     469.075
    drivewheel_rwd          -2774.1125   1804.172     -1.538      0.127   -6353.105     804.880
    enginelocation_front    -9804.5523   3424.429     -2.863      0.005   -1.66e+04   -3011.407
    enginelocation_rear       475.9443   2376.071      0.200      0.842   -4237.541    5189.430
    enginetype_dohc         -2561.2136   1254.359     -2.042      0.044   -5049.525     -72.902
    enginetype_dohcv        -2983.8772   3231.074     -0.923      0.358   -9393.458    3425.704
    enginetype_l              454.8178   2154.732      0.211      0.833   -3819.590    4729.226
    enginetype_ohc          -2433.7328   1069.723     -2.275      0.025   -4555.776    -311.690
    enginetype_ohcf         -4694.3952   2117.007     -2.217      0.029   -8893.967    -494.823
    enginetype_ohcv         -3084.0680   1692.623     -1.822      0.071   -6441.777     273.641
    enginetype_rotor         5973.8610   2562.832      2.331      0.022     889.892    1.11e+04
    fuelsystem_1bbl         -3329.5027   1541.659     -2.160      0.033   -6387.740    -271.266
    fuelsystem_2bbl          -378.8910    854.591     -0.443      0.658   -2074.170    1316.388
    fuelsystem_4bbl         -2935.9990   1970.721     -1.490      0.139   -6845.379     973.381
    fuelsystem_idi           1201.1596   2927.961      0.410      0.683   -4607.127    7009.447
    fuelsystem_mfi          -1366.8238   1996.130     -0.685      0.495   -5326.608    2592.961
    fuelsystem_mpfi         -1026.5666    871.045     -1.179      0.241   -2754.485     701.352
    fuelsystem_spdi         -1491.9845   1199.009     -1.244      0.216   -3870.495     886.526
    fuelsystem_spfi          4.089e-11   1.92e-11      2.134      0.035    2.87e-12    7.89e-11
    CompanyName_Nissan      -1803.3942   1835.116     -0.983      0.328   -5443.771    1836.983
    CompanyName_alfa-romero  1.197e+04   6316.252      1.895      0.061    -559.182    2.45e+04
    CompanyName_audi         1.183e+04   5797.196      2.041      0.044     330.786    2.33e+04
    CompanyName_bmw          1.573e+04   5206.982      3.021      0.003    5400.665    2.61e+04
    CompanyName_buick        9128.6993   2706.135      3.373      0.001    3760.455    1.45e+04
    CompanyName_chevrolet    4632.3280   4232.228      1.095      0.276   -3763.274     1.3e+04
    CompanyName_dodge        4460.6777   4031.469      1.106      0.271   -3536.672    1.25e+04
    CompanyName_honda        6598.5654   3476.301      1.898      0.061    -297.480    1.35e+04
    CompanyName_isuzu        3745.6382   2819.936      1.328      0.187   -1848.356    9339.632
    CompanyName_jaguar       8530.0230   4620.954      1.846      0.068    -636.707    1.77e+04
    CompanyName_maxda        2455.6772   2461.927      0.997      0.321   -2428.124    7339.479
    CompanyName_mazda        3315.7900   2025.752      1.637      0.105    -702.758    7334.338
    CompanyName_mercury      1679.6001   2685.979      0.625      0.533   -3648.660    7007.861
    CompanyName_mitsubishi  -2074.4582   1048.436     -1.979      0.051   -4154.273       5.356
    CompanyName_nissan      -2058.3218    926.838     -2.221      0.029   -3896.920    -219.724
    CompanyName_peugeot     -7526.2459   2604.698     -2.889      0.005   -1.27e+04   -2359.225
    CompanyName_plymouth    -6433.3823   2186.115     -2.943      0.004   -1.08e+04   -2096.719
    CompanyName_porcshce             0          0        nan        nan           0           0
    CompanyName_porsche      3549.2676   2719.076      1.305      0.195   -1844.647    8943.182
    CompanyName_renault     -4362.0171   2921.656     -1.493      0.139   -1.02e+04    1433.762
    CompanyName_saab        -1210.0138   2955.158     -0.409      0.683   -7072.253    4652.225
    CompanyName_subaru      -5170.3395   2157.428     -2.397      0.018   -9450.097    -890.583
    CompanyName_toyota      -9606.8477   4393.692     -2.187      0.031   -1.83e+04    -890.944
    CompanyName_toyouta      -1.14e+04   4894.113     -2.328      0.022   -2.11e+04   -1686.799
    CompanyName_vokswagen   -1.075e+04   6153.548     -1.748      0.084    -2.3e+04    1452.898
    CompanyName_volkswagen   -1.17e+04   5954.285     -1.965      0.052   -2.35e+04     113.070
    CompanyName_volvo       -1.099e+04   6029.573     -1.823      0.071    -2.3e+04     966.440
    CompanyName_vw          -1.187e+04   6353.808     -1.868      0.065   -2.45e+04     735.779
    ==============================================================================
    Omnibus:                       11.837   Durbin-Watson:                   2.019
    Prob(Omnibus):                  0.003   Jarque-Bera (JB):               18.463
    Skew:                           0.385   Prob(JB):                     9.79e-05
    Kurtosis:                       4.452   Cond. No.                     1.65e+16
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.99e-23. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    
