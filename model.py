#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

#Importing data set
dt=pd.read_csv('usa.csv')
x=dt.iloc[:,1:]
y=dt.iloc[:,0]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer1 = imputer.fit(x.iloc[0:,0:4])
x.iloc[0:,0:4] = imputer1.transform(x.iloc[0:,0:4])

imputer=SimpleImputer(missing_values=np.nan , strategy='mean')
imputer1=imputer.fit(x.iloc[:,:])
x.iloc[:,:]=imputer1.transform(x.iloc[:,:])

x["Rank"]=x['rank']
x.drop(['rank'],axis='columns')

outlier_col=['gre','gpa','ses','Race']
x.drop("rank",inplace=True,axis=1)
data=x.copy()

x['admit']=y.values

data=x.copy()
def z(x):
    outliers=[]
    mean=np.mean(x)
    std=np.std(x)
    s=3
    for i in x:
        z_score=(i - mean)/std
        if np.abs(z_score) > s:
            outliers.append(i)
    return outliers
for i in outlier_col:
    x1=data[i].values
    x1=z(x1)
    data=data[(data[i] != x1[0])]
    
x=data.iloc[:,:6]
y=data.iloc[:,6:]

data=x.copy()
lis=list(data.columns)
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scale.fit(data[lis])
data=pd.DataFrame(scale.transform(data[lis]),columns=lis)

x=data.copy()
os=RandomOverSampler()
X,Y=os.fit_resample(x,y)

clf=GridSearchCV(SVC(gamma=10),{
    'C':[0.5,1,5],
    "kernel":['rbf','linear']
},cv=5,
return_train_score=False)
clf.fit(X,Y)

df=pd.DataFrame(clf.cv_results_)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# intializing the clf to SVC
clf=SVC(kernel='rbf',C=5,gamma=10)

#fiting the data
clf.fit(x_train,y_train)

# predicting the sample data
sample=[520,2.93,3,1,2,4]
pred=clf.predict(scale.transform([sample]))

print(pred[0])
if pred[0]==1:
    print("YES")
else:
    print("NO")
    
import pickle
pickle.dump(clf,open('model.pkl','wb'))
mod=pickle.load(open('model.pkl','rb'))