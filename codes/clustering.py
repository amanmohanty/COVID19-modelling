from COVID_cluster import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_name):

	'''
	Read data from the csv files into pandas dataframe and merge data based on city

	'''
	us_states = pd.read_csv('../data/'+file_name[0])
	statelatlong = pd.read_csv('../data/'+file_name[1])
	data = pd.merge(us_states, statelatlong[['Latitude','Longitude','City']], on = 'City', how= 'left')
	return data

def sample_data(data,attribute,value):
	'''
	Get the data for the given attribute and value

	'''
	sampled_data = data.loc[data[attribute] == value]
	return sampled_data
	

def main():

	np.random.seed(6)  	# set random generator to reproduce results
	try_clusters = 50
	K = 15
	file_name = ["us_states.csv","statelatlong.csv"]
	attribute_to_query = 'date'
	value_to_query = '2020-03-25'

	# Read data from files

	data = read_data(file_name = file_name)
	print("\nFirst five rows of merged dataframe:\n",data.head())

	# Sample data based on Query

	sampled_data = sample_data(data=data, attribute=attribute_to_query, value=value_to_query)
	print("\nFirst five rows of sampled data:\n",sampled_data.head())

	# Run Unweighted K-means on sampled data

	unweighted_model = Unweighted_Cluster(n_clusters=K)
	inertia_elbow_unweighted = unweighted_model.elbow(data=sampled_data, try_clusters=try_clusters)
	cluster_set = unweighted_model.fit_predict()
	print("\nClusters with Unweighted K-means clustering\n")
	for cluster in range(K):
		print("\n Cluster-", cluster+1, end=" ")
		print(": ",cluster_set[cluster])

	# Run Weighted K-means on sampled data

	weighted_model = Weighted_Cluster(n_clusters=K)
	inertia_elbow_weighted = weighted_model.elbow(data=sampled_data, try_clusters=try_clusters)
	cluster_set = weighted_model.fit_predict()
	print("\nClusters with Weighted K-means clustering\n")
	for cluster in range(K):
		print("\n Cluster-", cluster+1, end=" ")
		print(": ",cluster_set[cluster])

	# Plot for COVID data scatter

	plt.figure(1)
	plt.scatter(sampled_data['Longitude'], sampled_data['Latitude'], sampled_data['cases']*0.3, c=np.random.rand(len(sampled_data['Latitude'])), alpha=0.5)
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.title('COVID data scatter plot')

	# Plots for Unweighted K-means

	plt.figure(2)
	plt.plot(list(range(1,try_clusters+1)), inertia_elbow_unweighted, linestyle='-', marker='.')
	plt.xlabel('Clusters')
	plt.ylabel('Cost value')
	plt.title('Cost v Number of Clusters plot')

	plt.figure(3)
	plt.scatter(sampled_data['Longitude'], sampled_data['Latitude'], sampled_data['cases']*0.3, c=np.random.rand(len(sampled_data['Latitude'])), alpha=0.5)
	plt.scatter(unweighted_model.cluster_centers[:,1],unweighted_model.cluster_centers[:,0])
	plt.xlabel('Longitude')
	plt.ylabel('Longitude')
	plt.title('COVID data scatter plot with unweighted cluster centers')

	# Plots for Weighted K-means

	plt.figure(4)
	plt.plot(list(range(1,try_clusters+1)), inertia_elbow_weighted, linestyle='-', marker='.')
	plt.xlabel('Clusters')
	plt.ylabel('Cost value')
	plt.title('Cost v Number of Clusters plot')

	plt.figure(5)
	plt.scatter(sampled_data['Longitude'], sampled_data['Latitude'], sampled_data['cases']*0.3, c=np.random.rand(len(sampled_data['Latitude'])), alpha=0.5)
	plt.scatter(weighted_model.cluster_centers[:,1],weighted_model.cluster_centers[:,0])
	plt.xlabel('Longitude')
	plt.ylabel('Longitude')
	plt.title('COVID data scatter plot with weighted cluster centers')


	plt.show()

if __name__ == '__main__':
	main()