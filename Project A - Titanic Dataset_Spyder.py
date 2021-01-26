#Importing required libraries

#Libraries for data processing
import pandas as pd
import numpy as np

#Libraries for Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline

# Module Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#Getting data
data_df=pd.read_csv('C:/Users/Muthu/Project A - Titanic/train.csv')
# get overall info for the dataset 
data_df.info()
data_df.describe()
data_df.shape


#Displaying total no. of null values
total1=data_df.isnull().sum()
total2=data_df.isnull().count()
Missing_value=round(((total1/total2)*100),2)
Missing_value_df=pd.concat([total1, Missing_value], axis=1, keys=['Total', '%'])
Missing_value_df

#Displaying correlation for each variable
data_df.corr(method='pearson')

#Gender Vs Survival - Countplot
sns.barplot(x="Sex", y="Survived", data=data_df).set_title('Gender Vs Survival')
print("Percentage of females who survived:", data_df["Survived"][data_df["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", data_df["Survived"][data_df["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#The above Countplot shows that Female has number of survival.

#Survival by AGE - Histogram
age=data_df['Age'][data_df.Age.notnull()] 
plt.hist(age ,20, histtype='stepfilled')
plt.ylabel('Frequncy')
plt.title('Age Distrubition Histogram')
plt.xlabel('Age')

#Pclass Vs Survival - Barplot
sns.barplot(x='Pclass', y='Survived', data=data_df).set_title('Pclass Vs Survival')
#print percentages of females vs. males that survive
print("Percentage of Class1 who survived:", data_df["Survived"][data_df["Pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Class2 who survived:", data_df["Survived"][data_df["Pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Class3 who survived:", data_df["Survived"][data_df["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#The above plot clearly shows that Pclass 1 has the most survival comparing to the other 2 class.

#Parch Vs Survival - Barplot
sns.barplot(x='Parch', y='Survived', data=data_df).set_title('Parch Vs Survival')

#Data Preprocessing
#By looking at the variables we will not require PassengerId hence will drop it.
data_df=data_df.drop(['PassengerId'],axis=1)

data_df['Ticket'].describe()
data_df['Cabin'].describe()
#It seens we have lots of unique values in both Ticket and Cabin variable, so it'll be tricky to use this variable. Hence we will drop it.
data_df=data_df.drop(['Ticket'],axis=1)
data_df=data_df.drop(['Cabin'],axis=1)

#Missing value interpretation
#Embarked
print('Number of people embarking in :',data_df['Embarked'].value_counts())
#We can add missing value as S, since we have most occurence value S.
data_df=data_df.fillna({"Embarked": "S"})

#Age
Mis_Age=data_df['Age'].isnull().sum()
mean=data_df['Age'].mean()
rand_age=np.random.randint(mean-10,mean+10,Mis_Age)
age_slice = data_df["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data_df["Age"] = age_slice
data_df["Age"] = data_df["Age"].astype(int)
data_df["Age"]

#Converting features
#Sex
Gender = {'male' : 0, 'female' : 1}
data_df['Gender']=data_df['Sex'].map(Gender)
data_df['Gender']


#Embarked
Embar_Mapping = {'S' : 0, 'C' : 1, 'Q' : 2}
data_df['Boarding'] = data_df['Embarked'].map(Embar_Mapping)
data_df['Boarding']

#Age
data_df['Age'] = data_df['Age'].astype(int)
data_df['Categories'] = data_df['Age']
data_df.loc[data_df['Age'] <= 14, 'Categories'] = 'Children'
data_df.loc[(data_df['Age'] >= 15) & (data_df['Age'] <= 24), 'Categories'] = 'Youth'
data_df.loc[(data_df['Age'] >= 25) & (data_df['Age'] <= 64), 'Categories'] = 'Adult'
data_df.loc[data_df['Age'] > 64, 'Categories'] = 'Senior'
data_df['Categories']
Age_Mapping = {'Children' : 0, 'Youth' : 1, 'Adult' : 2, 'Senior' : 3}
data_df['New_Age'] = data_df['Categories'].map(Age_Mapping)
data_df['New_Age']

#Fare
data_df['Fare'] = data_df['Fare'].astype(int)
data_df['Expense'] = data_df['Fare']
data_df.loc[data_df['Fare'] < 8, 'Expense'] = '0'
data_df.loc[(data_df['Fare'] >= 8) & (data_df['Fare'] <= 15), 'Expense'] = '1'
data_df.loc[(data_df['Fare'] >= 16) & (data_df['Fare'] <= 30), 'Expense'] = '2'
data_df.loc[(data_df['Fare'] >= 31) & (data_df['Fare'] <= 100), 'Expense'] = '3'
data_df.loc[(data_df['Fare'] >= 101) & (data_df['Fare'] <= 250), 'Expense'] = '4'
data_df.loc[data_df['Fare'] > 250, 'Expense'] = '5'

#Name
data_df['Title'] = data_df['Name'].str.extract(' ([A-Za-z]+)\.' , expand=False)
data_df['Title'] = data_df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev'], 'Rare')
data_df['Title'] = data_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
data_df['Title'] = data_df['Title'].replace(['Mlle'], 'Miss')
data_df['Title'] = data_df['Title'].replace(['Mme'], 'Mrs')
data_df['Title'] = data_df['Title'].replace(['Ms'], 'Miss')
pd.crosstab(data_df['Title'], data_df['Sex'])

Title_conv = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Royal': 4, 'Rare': 5}
data_df.Title = data_df.Title.map(Title_conv)
data_df['Title']

#Dropping the remaining variables after conversion
data_df = data_df.drop(['Sex'], axis=1)
data_df = data_df.drop(['Embarked'], axis=1)
data_df = data_df.drop(['Name'], axis=1)
data_df = data_df.drop(['Fare'], axis=1)
data_df = data_df.drop(['Age'], axis=1)
data_df = data_df.drop(['Categories'], axis=1)

data_df.head(10)

#Splitting Train and Test dataset
from sklearn.model_selection import train_test_split
predictors = data_df.drop('Survived', axis=1)
target = data_df['Survived']

xtrain, xtest, Ytrain, Ytest = train_test_split(predictors, target, test_size=0.30, random_state=0)

#Models
from sklearn.metrics import accuracy_score

#Logistic Regression
LogReg = LogisticRegression()
LogReg.fit(xtrain, Ytrain)
y_pred = LogReg.predict(xtest)
Acc_LogReg = round(accuracy_score(y_pred,Ytest)*100, 2)
Acc_LogReg

#Decision Tree
Decision_Tree = DecisionTreeClassifier()
Decision_Tree.fit(xtrain, Ytrain)
y_pred = Decision_Tree.predict(xtest)
Acc_Decision_Tree = round(accuracy_score(y_pred,Ytest)*100, 2)
Acc_Decision_Tree

#Random Forest
randomforest = RandomForestClassifier()
randomforest.fit(xtrain, Ytrain)
y_pred = randomforest.predict(xtest)
Acc_randomforest = round(accuracy_score(y_pred, Ytest)*100, 2)
Acc_randomforest

#K Nearest Neighbor
knn = KNeighborsClassifier()
knn.fit(xtrain, Ytrain)
y_pred = knn.predict(xtest)
Acc_Knn = round(accuracy_score(y_pred, Ytest)*100, 2)
Acc_Knn

#Naive Bayes
Gaussian = GaussianNB()
Gaussian.fit(xtrain, Ytrain)
y_pred = Gaussian.predict(xtest)
Acc_Gaussian = round(accuracy_score(y_pred, Ytest)*100, 2)
Acc_Gaussian

#Showing all models accuracy
Result = pd.DataFrame({
        'Models' : ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Gaussian'],
        'Score' : [Acc_LogReg, Acc_Decision_Tree, Acc_randomforest, Acc_Knn, Acc_Gaussian]
        })
Result.sort_values(by='Score', ascending=False)

#Importance of each features under Decision Tree
Importance_Feature = pd.DataFrame({
        'Feature' : xtrain.columns, 
        'Importance' : np.round(randomforest.feature_importances_,3)
            })
Importance_Feature.sort_values(by='Importance', ascending=False)



