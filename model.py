import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split

cancer = pd.read_csv('breast_cancer.csv')
cancer = cancer.iloc[:, 1:32]

cancer['diagnosis'].replace("M","0",inplace=True)
cancer['diagnosis'].replace("B","1",inplace=True)
cancer['diagnosis'] = cancer['diagnosis'].astype(str).astype(int)

x = cancer.drop(['diagnosis'], axis = 1)
y = cancer['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

svm = SVC(C= 10, gamma= 0.01, kernel= 'rbf')
LR = LogisticRegression(max_iter=3000,C= 1.0, penalty= 'l2', solver= 'liblinear')

est = [('svm',svm), ('lr',LR)]
classifier = VotingClassifier(estimators = est, voting ='hard')

classifier.fit(x_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))
