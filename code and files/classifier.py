import sys
import time
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import f1_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

class Classifier:

	def __init__(self):
		self.case = 1
		self.LOOPS = 1
		self.train_size = 0.75
		self.n_estimators = 10
		self.topK = 8
		self.df = pd.read_csv("case_{}.csv".format(self.case))
		self.df = self.df.iloc[:, 1:]
		self.classify()

	def get_downsampled_df(self):
		if self.case == 1:
			df_majority = self.df[self.df["class"]==0]
			df_minority = self.df[self.df["class"]==1]
		else:
			df_majority = self.df[self.df["class"]==1]
			df_minority = self.df[self.df["class"]==0]

		df_majority_downsampled = resample(df_majority, 
		                                 replace=False,   
		                                 n_samples=len(df_minority["class"]))
		df_downsampled = pd.concat([df_majority_downsampled, df_minority])
		return df_downsampled

	def classify(self):
		avg_f1, avg_accuracy = 0.0, 0.0

		for i in range(self.LOOPS):
			f1, accuracy = self.classify_iteration()
			avg_f1 += f1
			avg_accuracy += accuracy
			

		avg_f1 /= self.LOOPS
		avg_accuracy /= self.LOOPS
		avg_f1 = round(avg_f1, 2)
		avg_accuracy = round(avg_accuracy, 2)

		print ("F1-score:", avg_f1)
		print("Accuracy:", avg_accuracy)

	def classify_iteration(self):
		df = self.get_downsampled_df()
		X, y = self.feature_selection(df)
		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)		
		clf = ExtraTreesClassifier(n_estimators=self.n_estimators)
		clf = clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)	
		return f1_score(y_test, y_pred, average='macro'), accuracy_score(y_test, y_pred)
		
	def feature_selection(self, df):
		X, y = df.iloc[:, :-1], df.iloc[:, -1]
		clf = ExtraTreesClassifier(n_estimators=self.n_estimators)
		clf = clf.fit(X, y)

		d = {}
		selected_features = []
		pos = 0
		for i in clf.feature_importances_:
			d[i]=pos
			pos += 1
		for i in sorted(d.keys())[::-1][:self.topK]:
			selected_features.append(d[i])

		selected_columns = [X.columns[i] for i in selected_features]
		X = X[selected_columns]

		return X, y

start_time = time.time()
classifier = Classifier()
print ("TIME:", time.time() - start_time)