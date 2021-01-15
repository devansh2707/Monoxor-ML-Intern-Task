import pandas as pd
import numpy as np
import pickle
import re
import xlrd
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df=pd.read_csv(r'C:\Users\DELL\file.csv')
df.head()
df.rename(columns = {'req/body/note/desc':'COMMENTS'}, inplace = True) 
ps=PorterStemmer()
corpus=[]

def clearning():                 #function used for cleaning text
    for i in range(len(df)):
        review=re.sub('[^a-zA-Z]',' ',df['COMMENTS'][i])
        review=review.lower()
        review=review.split()
        review=[ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        corpus.append(review)
clearning()

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer() 
X=cv.fit_transform(corpus).toarray()
y=df['isSafe']
print(X)
y = y.astype('int')
m = y.dtype

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=0)
    
sc=StandardScaler()    


def fitting(X_train, X_test ):
    #print("hello")
    
    #sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)

fitting(X_train, X_test)
nb = GaussianNB()
    
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
from sklearn.metrics import classification_report
report=classification_report( y_test, y_pred)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 

print ("Confusion Matrix : \n", cm) 

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred)) 