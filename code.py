#import necissar library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv('Social_Network_Ads.csv')
#discover data
print(df.head())
print("data discription\n")
print(df.shape)
print(df.columns)
df.drop(columns=['User ID'],axis=1,inplace=True)
print(df.describe())
print(df.info())
#x=df['Gender']
#y=df['Purchased']

plt.hist(df['Gender'])
plt.title("figure 1")
plt.xlabel("gender")
plt.show()
plt.hist(df['EstimatedSalary'])
plt.title("figure 2")
plt.xlabel("Estimatedsalary")
plt.show()
print(pd.crosstab(df.Gender,df.Purchased))
#changing data
df['Gender'].replace(['Female','Male'],[0,1],inplace=True)


#draw correlation matrix
corrmat = df.corr()
plt.subplots(figsize=(10,6))
sns.heatmap(corrmat,annot=True,linewidth=0.5,fmt=".2f", cmap="viridis")
plt.title("figure 3")
plt.show()
print(corrmat)

#define variabeles
y=df['Purchased']
x=df.drop(['Purchased'],axis=1)
#une courbe qui relie le nombre de voisins (K) et le score du modèle
error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x,y)
    pred = knn.predict(x)
    error_rate.append(np.mean(pred != y))

plt.figure(figsize=(15,10))
plt.plot(range(1,50),error_rate, marker='o', markersize=9)
plt.show()
#........
#we will take the optimal value of n_neighbors
#Instancier le modèle
model=KNeighborsClassifier(n_neighbors=1)
#Entrainer votre modèle
model.fit(x,y)
print(model.score(x,y))

#************************************************
#draw correlation matrix
corrmat = df.corr()
plt.subplots(figsize=(10,6))
sns.heatmap(corrmat,annot=True,linewidth=0.5,fmt=".2f", cmap="viridis")
plt.title("figure 3")
plt.show()
print(corrmat)

#define variabeles
y=df['Survived']
x=(df[['Fare','Parch','SibSp']])
#une courbe qui relie le nombre de voisins (K) et le score du modèle
error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x,y)
    pred = knn.predict(x)
    error_rate.append(np.mean(pred != y))

plt.figure(figsize=(15,10))
plt.plot(range(1,50),error_rate, marker='o', markersize=9)
plt.show()
#........
#we will take the optimal value of n_neighbors
#Instancier le modèle
model=KNeighborsClassifier(n_neighbors=1)
#Entrainer votre modèle
model.fit(x,y)
print("knn score\t")
print(model.score(x,y))
print("\n********\n")
model2 = LogisticRegression()
model2.fit(x,y)
print("logistic regressor score\t")
print(model2.score(x,y))
print("\n********\n")
