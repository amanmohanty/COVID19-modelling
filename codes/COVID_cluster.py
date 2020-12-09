
import numpy as np
from sklearn.cluster import KMeans
np.random.seed(6)

class Cluster(object):

	'''
	Model to implement k-means clustering for 2-dimensional COVID data
	'''

	def __init__(self, n_clusters, weight):
		'''
		Initialise K-means
		n_clusters: Number of cluster for K-means
		weight: (bool) True - weighted clustering, False - unweighted clustering
		'''
		self.inertia_elbow = []					
		self.clusters = n_clusters
		self.labels = None
		self.cluster_centers = None
		self.kmeans = KMeans(n_clusters = self.clusters, init = 'k-means++', random_state=6)
		self.weight = weight
		self.sample_weight = None


	def elbow(self, data, try_clusters, data_loc):
		'''
		Use Elbow method to find variation between cost and clusters
		data_loc: the location of the column of the first data in pandas dataframe; data must be in 
		consecutive columns
		try_clusters: Number of clusters to try Elbow method

		If self.weight is True store sample weight 
		'''
		self.org_data = data.dropna()
		self.data = self.org_data[self.org_data.columns[data_loc:]]
		if self.weight == True:
			self.sample_weight = self.org_data[self.org_data.columns[3]]
		for k in range(1,try_clusters+1):
			try_model = KMeans(n_clusters = k, init = 'k-means++',random_state=6)
			try_model.fit(self.data,sample_weight=self.sample_weight)
			self.inertia_elbow.append(try_model.inertia_)

		return self.inertia_elbow

	def fit_predict(self,data_loc):
		'''
		Perform k-means on the sampled data and create cluster set of cities belonging to a cluster
		data_loc: location of the column 'City'
		'''
		self.labels = self.kmeans.fit_predict(self.data,sample_weight=self.sample_weight)
		self.cluster_centers = self.kmeans.cluster_centers_
		cluster_set = [0]*self.clusters
		for cluster in range(self.clusters):
			temp = np.array(np.where(self.labels==cluster)).flatten()   
			cluster_set[cluster] = self.org_data.iloc[temp,data_loc].tolist()
		return cluster_set


class Unweighted_Cluster(object):

	'''
	Model to implement unweighted k-means clustering
	'''

	def __init__(self, n_clusters):
		'''
		Initialise K-means and do elbow method
		'''
		self.inertia_elbow = []					
		self.clusters = n_clusters
		self.labels = None
		self.cluster_centers = None
		self.kmeans = KMeans(n_clusters = self.clusters, init = 'k-means++')


	def elbow(self, data, try_clusters):
		'''
		Use Elbow method and plot variation of cost with clusters
		'''
		self.org_data = data.dropna()
		self.data = self.org_data[self.org_data.columns[5:]]
		for k in range(1,try_clusters+1):
			try_model = KMeans(n_clusters = k, init = 'k-means++')
			try_model.fit(self.data)
			self.inertia_elbow.append(try_model.inertia_)

		return self.inertia_elbow

	def fit_predict(self):
		'''
		Perform k-means on the sampled data and print the names of cities belonging to the same cluster
		'''
		self.labels = self.kmeans.fit_predict(self.data)
		self.cluster_centers = self.kmeans.cluster_centers_
		cluster_set = [0]*self.clusters
		for cluster in range(self.clusters):
			temp = np.array(np.where(self.labels==cluster)).flatten()   # Find location of the labels where labels is equal to clusters
			cluster_set[cluster] = self.org_data.iloc[temp,1].tolist()
		return cluster_set

class Weighted_Cluster(object):

	'''
	Model to implement unweighted k-means clustering
	'''

	def __init__(self, n_clusters):
		'''
		Initialise K-means and do elbow method
		'''
		self.inertia_elbow = []					
		self.clusters = n_clusters
		self.labels = None
		self.cluster_centers = None
		self.kmeans = KMeans(n_clusters = self.clusters, init = 'k-means++')


	def elbow(self, data, try_clusters):
		'''
		Use Elbow method and plot variation of cost with clusters
		'''
		self.org_data = data.dropna()
		self.data = self.org_data[self.org_data.columns[5:]]
		for k in range(1,try_clusters+1):
			try_model = KMeans(n_clusters = k, init = 'k-means++')
			try_model.fit(self.data)
			self.inertia_elbow.append(try_model.inertia_)

		return self.inertia_elbow

	def fit_predict(self):
		'''
		Perform k-means on the sampled data and print the names of cities belonging to the same cluster
		'''
		self.labels = self.kmeans.fit_predict(self.data)
		self.cluster_centers = self.kmeans.cluster_centers_
		cluster_set = [0]*self.clusters
		for cluster in range(self.clusters):
			temp = np.array(np.where(self.labels==cluster)).flatten()   # Find location of the labels where labels is equal to clusters
			cluster_set[cluster] = self.org_data.iloc[temp,1].tolist()
		return cluster_set


