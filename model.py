import pandas as pd
import numpy as np
import matplotlib.pyplot as py
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler



# Importing data set
dt=pd.read_csv('usa.csv')
print(dt)
x=dt.iloc[:,1:]
y=dt.iloc[:,0]

# Data preprocessing operations
# 1) Finding and managing the missing vlaues
# x.isnull().sum()
# gre            2
# gpa            1
# ses            1
# Gender_Male    1
# Race           0
# rank           1
# dtype: int64


# perfomring operations from columns 0 to 3
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer1 = imputer.fit(x.iloc[0:,0:4])
x.iloc[0:,0:4] = imputer1.transform(x.iloc[0:,0:4])


# performing operations(missing values) on the rank column
imputer=SimpleImputer(missing_values=np.nan , strategy='mean')
imputer1=imputer.fit(x.iloc[:,:])
x.iloc[:,:]=imputer1.transform(x.iloc[:,:])

# 2) finding and eliminating the outliers
# we can do this by using IQR or Z-score methods
#changing Rank from rank because x have the pre-defined method name rank and there is a possibilty of getting an error
x["Rank"]=x['rank']
x.drop(['rank'],axis='columns')

outlier_col=['gre','gpa','ses','Race']
x['admit']=y.values

# using z-score method
# formula z=(i-mean)/std
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
    x1=x[i].values
    x1=z(x1)
    x=x[(x[i] != x1[0])]
x.drop(['rank'],axis='columns')
y=x.iloc[:,7]

# checking the data is balanced or not
# we can see that the data is imbalanced and we can make it a balanced dataset 
# we can fix this by performing undersampling,oversampling,SMOTETotek 
# And I am performing oversampling technique to balance the data
os=RandomOverSampler()
X,Y=os.fit_resample(x,y)
X1=X.drop(['admit'],axis='columns')
X1=X1.drop(['rank'],axis='columns')
Y1=Y

# Building the model (SVC)
# before building the model we have to know the perfect hyperparameters to use
# so that we can get a good accuracy . For that we are using HYPERPARAMETER TUNNING namely GridSearchCV¶
#internally GridSearchCV use the k-fold technique for training purpose
clf=GridSearchCV(SVC(gamma=10),{
    'C':[0.5,1,5],
    "kernel":['rbf','linear']
},cv=5,
return_train_score=False)
clf.fit(X1,Y1)

# df=pd.DataFrame(clf.cv_results_)
# df[['param_C','param_kernel','mean_test_score']]

# param_C	param_kernel	mean_test_score
#  0.5	        rbf	          0.714169
#  0.5	       linear	      0.636442
# 	1	        rbf	          0.898760
# 	1	       linear         0.619878
# 	5	        rbf           0.898760
# 	5	       linear         0.621696

#here we are spliting our dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(X1,Y1,test_size=0.2,random_state=0)

# after knowing the hyperparameters from the GridSearchCV
# we are going to build the model
# intializing the clf to SVC
clf=SVC(kernel='rbf',C=5,gamma=10)

#fiting the data
clf.fit(x_train,y_train)

# testing the accuracy of the model
# print(clf.score(x_test,y_test))
# 0.8440366972477065

# predicting the sample data
# pred=clf.predict([[640.0,3.19,1.0,1.0,2.0,4.0]])
# if pred==y[1]:
#     print("YES")
# else:
#     print("NO")
# ans is YES

# saving model to disk
import pickle
pickle.dump(clf,open('model.pkl','wb'))

mod=pickle.load(open('model.pkl','rb'))
pred=mod.predict([[640.0,3.19,1.0,1.0,2.0,4.0]])
if pred==y[1]:
    print("YES")
else:
    print("NO")

print(np.dtype(y[1]),y[1])