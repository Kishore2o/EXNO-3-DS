## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="575" height="691" alt="image" src="https://github.com/user-attachments/assets/fb8829bd-4412-4b64-8b3a-37e2e8887858" />


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="716" height="365" alt="image" src="https://github.com/user-attachments/assets/1f727796-02fc-4440-ae80-fe7b4fb85327" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="606" height="585" alt="image" src="https://github.com/user-attachments/assets/bfbc3d10-be65-4f3d-b0c1-78dbf432b636" />


```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
<img width="511" height="636" alt="image" src="https://github.com/user-attachments/assets/d0d1eeec-e81f-4f35-a44d-d24570187c2c" />


```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

ohe = OneHotEncoder(sparse_output=False)  # use sparse_output instead of sparse
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]), 
                   columns=ohe.get_feature_names_out(["nom_0"]))
df2 = pd.concat([df2, enc], axis=1)
print(df2)
```
<img width="866" height="516" alt="image" src="https://github.com/user-attachments/assets/55f7d85b-3508-4fac-a62a-f20d036e2773" />


```
 pd.get_dummies(df2,columns=["nom_0"])
```
<img width="1133" height="515" alt="image" src="https://github.com/user-attachments/assets/7bdc05d9-f103-41e0-b154-151c0bff27c1" />

```
pip install --upgrade category_encoders
```
<img width="1352" height="506" alt="image" src="https://github.com/user-attachments/assets/bc020e09-76b3-4010-8fa6-ebcb7b5003a3" />

```
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
```
<img width="585" height="149" alt="image" src="https://github.com/user-attachments/assets/1f38287b-8933-4665-b6a0-415c176049b2" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="687" height="605" alt="image" src="https://github.com/user-attachments/assets/801f1596-9450-40b8-96bc-df4ce7b87825" />


```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="963" height="586" alt="image" src="https://github.com/user-attachments/assets/f9b1f5bd-6760-4998-8db2-711ce370ca42" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="827" height="612" alt="image" src="https://github.com/user-attachments/assets/6ee1b872-5675-4937-ad9d-ad6da483da38" />


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1096" height="717" alt="image" src="https://github.com/user-attachments/assets/2eb5127e-da92-4b14-bf81-a9ebdbb42078" />

```
 df.skew()
```
<img width="465" height="315" alt="image" src="https://github.com/user-attachments/assets/9394ea2d-a835-4fc4-92c2-213f66a11d46" />


```
np.log(df["Highly Positive Skew"])
```
<img width="416" height="623" alt="image" src="https://github.com/user-attachments/assets/455206f6-722d-4a99-8b67-ae7bd8d7effd" />


```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="507" height="647" alt="image" src="https://github.com/user-attachments/assets/e0f74905-2fae-453e-99c3-2372e3137bab" />


```
np.sqrt(df["Highly Positive Skew"])
```
<img width="446" height="622" alt="image" src="https://github.com/user-attachments/assets/6ed4c6f5-73cd-48df-b4b6-4282297c094a" />


```
 np.square(df["Highly Positive Skew"])
```

<img width="443" height="623" alt="image" src="https://github.com/user-attachments/assets/5cde027e-eb1d-4e2c-ac07-11051f8f4915" />


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1312" height="607" alt="image" src="https://github.com/user-attachments/assets/727539c4-f98f-4cf2-a3a7-fcfe7d98cd05" />

```
df.skew()
```

<img width="577" height="402" alt="image" src="https://github.com/user-attachments/assets/dd6a7bf4-1caa-4529-8e92-9184c5442020" />


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="977" height="410" alt="image" src="https://github.com/user-attachments/assets/73ed3f9e-b491-4d35-b429-be43bb7b3b0c" />


```
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal', random_state=0)
df["Moderate_Negative_Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df

```
<img width="1192" height="710" alt="image" src="https://github.com/user-attachments/assets/27b845ca-1e71-4449-8d07-b23a5776865a" />


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="897" height="701" alt="image" src="https://github.com/user-attachments/assets/4d18ca22-413f-40c7-9836-07af2c9b8dd8" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="805" height="640" alt="image" src="https://github.com/user-attachments/assets/eb449726-417c-4f21-9ea3-8987108b88dd" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="826" height="698" alt="image" src="https://github.com/user-attachments/assets/78ac5ba3-8585-49a1-8232-a6ac285caf28" />





```



```
# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file
  was performed successfully

       
