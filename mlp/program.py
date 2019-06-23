import pandas as pd
import sys
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

def linear_regression(X_train, X_test, y_train, y_test):
	lm = linear_model.LinearRegression()
	lm_model = lm.fit(X_train, y_train)
	lm_predictions = lm.predict(X_test)
	print('Linear Regression score is %f' % lm.score(X_test, y_test))
    
def lasso_regression(X_train, X_test, y_train, y_test):
	ls = linear_model.Lasso(max_iter=2000)
	ls_model = ls.fit(X_train, y_train)
	ls_predictions = ls.predict(X_test)
	print('Lasso Regression score is %f' % ls.score(X_test, y_test))

def ridge_regression(X_train, X_test, y_train, y_test):
	lr = linear_model.Ridge(max_iter=2000)
	lr_model = lr.fit(X_train, y_train)
	lr_predictions = lr.predict(X_test)
	print('Ridge Regression score is %f' % lr.score(X_test, y_test))

def main():
	url="https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv"
	df = pd.read_csv(url, index_col=0)
    
	# Renaming columns for easier identification
	variables = ['date', 'house_age', 'dist_to_mrt', 'num_conv_stores','lat', 'long']
	target = ['price_per_unit_area']

	df.columns = variables+target

	# Scale - normalize and center values
	names = df[variables].columns
	scaler = preprocessing.StandardScaler()
	scaled_df = scaler.fit_transform(df[variables])
	scaled_df = pd.DataFrame(scaled_df, columns=names)
    
	# Peform a train/test split of 80/20
	y = df[target]
	X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2)

	if (int(sys.argv[1]) == 1):
		print("Applying Linear Regression algorithm for prediction...")
		linear_regression(X_train, X_test, y_train, y_test) 
	elif (int(sys.argv[1]) == 2): 
		print("Applying Lasso Regression algorithm for prediction...")
		lasso_regression(X_train, X_test, y_train, y_test)
	elif (int(sys.argv[1]) == 3): 
		print("Applying Ridge Regression algorithm for prediction...")
		ridge_regression(X_train, X_test, y_train, y_test)
	elif (int(sys.argv[1]) == 4):
		print("End.")
	else:
		print("Invalid")    

if __name__ == '__main__':
	main()