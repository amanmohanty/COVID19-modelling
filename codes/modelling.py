import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from COVID_cluster import Cluster

class Exp_Model(object):

	'''
	Class to perform exponential modeling of COVID growth data
	'''

	def __init__(self,data):
		'''
		Initialise parameters to dictionaries

		'''
		self.data = data
		self.A = {}
		self.B = {}
		self.pred_cases = {}
		self.X = {}
		self.Y = {}

	def fit(self):
		'''
		Store list of unique cities and perform Least Square Poly Fit using numpy.polyfit()
		'''
		self.unique_city = self.data['City'].unique()
		for city in self.unique_city:
			temp_data = self.data.loc[self.data['City'] == city]
			self.X[city] = temp_data['date'].tolist()
			self.Y[city] = temp_data['cases'].tolist()
			coeff = np.polyfit(self.X[city],np.log(self.Y[city]),deg=1)
			self.A[city] = np.exp(coeff[1])
			self.B[city] = coeff[0]

	def predict(self):
		'''
		Based on the derived model parameters predict cases
		'''
		for city in self.unique_city:
			temp = [x * self.B[city] for x in self.X[city]]
			self.pred_cases[city] = self.A[city] * np.exp(temp)

def read_data(file_name):

	'''
	Read data from the csv files into pandas dataframe and modify

	'''
	us_states = pd.read_csv('../data/'+file_name)
	val = list(range(0,us_states.shape[0]))
	del us_states['date']
	us_states.insert(0,'date',pd.Series(val))

	return us_states


def main():

	file_name = "us_states.csv"
	data = read_data(file_name = file_name)

	exp_model = Exp_Model(data=data)
	exp_model.fit()
	exp_model.predict()

	plot_X_California = exp_model.X.get('California')
	plot_Y_California = exp_model.pred_cases.get('California')
	plot_X_NY = exp_model.X.get('New York')
	plot_Y_NY = exp_model.pred_cases.get('New York')

	plot_data_California = data.loc[data['City'] == 'California']
	plot_data_NewYork = data.loc[data['City'] == 'New York']


	# Get data from model dictionaries and create a new dataframe
	new_data = pd.DataFrame.from_dict(exp_model.A,orient='index',columns=['A'])
	new_data.reset_index(inplace=True)
	new_data.insert(2,'B',exp_model.B.values())

	# Cities with their A and B values

	params = {}
	for city in exp_model.A.keys():
		params[city]=(exp_model.A[city],exp_model.B[city])
		print(city, ' : ', params[city])

	# Run Unweighted K-means on data
	K = 8
	try_clusters = 50

	unweighted_model = Cluster(n_clusters=K, weight=False)
	inertia_elbow_unweighted = unweighted_model.elbow(data=new_data, try_clusters=try_clusters, data_loc=1)
	cluster_set = unweighted_model.fit_predict(data_loc=0)
	print("\nClusters with Un-weighted K-means clustering\n")
	for cluster in range(K):
		print("\n Cluster-", cluster+1, end=" ")
		print(": ",cluster_set[cluster])

	# Plot for COVID growth of California and New York
	
	plt.figure(1)
	plt.plot(plot_data_California['date'].tolist(),plot_data_California['cases'].tolist())
	plt.plot(plot_data_NewYork['date'].tolist(),plot_data_NewYork['cases'].tolist())
	plt.legend(('California','New York'))
	plt.xlabel('Time')
	plt.ylabel('Cases')
	plt.title('Trend of COVID cases with respect to time')

	#Plot for Poly Fit of COVID growth for all states
	
	plt.figure(2)
	plt.plot(plot_data_California['date'].tolist(),plot_data_California['cases'].tolist())
	plt.plot(plot_data_NewYork['date'].tolist(),plot_data_NewYork['cases'].tolist())
	plt.plot(plot_X_California,plot_Y_California)
	plt.plot(plot_X_NY,plot_Y_NY)
	plt.legend(('True California','True New York','Predicted California','Predicted New York'))
	plt.xlabel('Time')
	plt.ylabel('Cases')
	plt.title('Predicted COVID growth with respect to time')


	# Plot for Elbow

	plt.figure(3)
	plt.plot(list(range(1,try_clusters+1)), inertia_elbow_unweighted, linestyle='-', marker='.')
	plt.xlabel('Clusters')
	plt.ylabel('Cost value')
	plt.title('Cost v Number of Clusters plot')

	# Plot for Cluster Centers

	plt.figure(4)
	plt.scatter(new_data['A'], new_data['B'])
	plt.scatter(unweighted_model.cluster_centers[:,0],unweighted_model.cluster_centers[:,1], c='r')
	plt.ylim(0,0.01)
	plt.xlabel('A')
	plt.ylabel('B')
	plt.title('COVID data scatter plot for parameters with unweighted cluster centers')

	plt.show()

if __name__ == '__main__':
	main()