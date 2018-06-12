import caffe
import numpy as np
import os
import io
import sys
	
class DataAnalysisWeightedAverages(caffe.Layer):
	def setup(self, bottom, top):
		#Checking inputs and outputs
		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 2 features and 1 label')
		if len(top)!=1:
			raise Exception('one output: Mahalanobis Matrix')
		
		#Parameters reading	
		params = eval(self.param_str)
		self.beta = params["beta"]
       		self.snapshot = params["snapshot_step"]
		
		self.iteration = 0
                self.Y = np.zeros((bottom[0].num, 1), dtype=np.float32)				# Labels array	
		self.S = np.identity((bottom[0].channels), dtype=np.float32)			# Similarity covariance matrix
		self.D = np.identity((bottom[0].channels), dtype=np.float32)			# Dissimilarity covariance matrix
                self.Smean = np.zeros((bottom[0].channels), dtype=np.float32)
		self.Dmean = np.zeros((bottom[0].channels), dtype=np.float32)
		
	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')
		#output has Mahalanobis Matrix dimensions
                top[0].reshape(bottom[0].channels, bottom[0].channels)

		self.diff = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)     # Features difference has descriptors dimensions
                self.Smean_batch = np.zeros((bottom[0].channels), dtype=np.float32)			# Similarity expected values array
		self.Dmean_batch = np.zeros((bottom[0].channels), dtype=np.float32)			# Dissimilarity expedted values array
		self.Sbatch = np.identity((bottom[0].channels), dtype=np.float32)		# Similarity covariance matrix from the current batch
		self.Dbatch = np.identity((bottom[0].channels), dtype=np.float32)               # Dissimilarity covariance matrix from the current batch
		self.M = np.identity((bottom[0].channels), dtype=np.float32) 			# Mahalanobis matrix
		self.col= np.zeros((bottom[0].channels, 1), dtype=np.float32)
 		self.row= np.zeros((1, bottom[0].channels), dtype=np.float32)
		 	
		
	def forward(self, bottom, top):
		self.iteration=self.iteration+1
		self.Y[:,0] = bottom[2].data[...]
        	self.diff = bottom[0].data[...] - bottom[1].data[...] 

		
		#Get discriminative sets of difference feature
		Slist=[]							# Similarity list from the batch
		Dlist=[]							# Dissimilarity list from the batch
		for i in range(bottom[0].num):
			if self.Y[i] == 1.0 :
				Slist.append(self.diff[i,:])
			else:
				Dlist.append(self.diff[i,:])
                
		Ssize=len(Slist)						# Number of similarity elements
		Dsize=len(Dlist)						# Number of dissimilarity elements
                Sset = np.zeros((Ssize, bottom[0].channels), dtype=np.float32)	# Similarity set from the current batch
		Dset = np.zeros((Dsize, bottom[0].channels), dtype=np.float32)	# Dissimilarity set from the current batch

 		#Move set lists to sets arrays
		for s in range(Ssize):
			Sset[s,:] = Slist[s]
		for d in range(Dsize):
			Dset[d,:] = Dlist[d]
	
	
		Svar = np.zeros((Ssize, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Similarity variations array
		Dvar = np.zeros((Dsize, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Dissimilarity variations array
				
                # Similarity covariance matrix computation from the current batch
		self.Smean_batch = np.sum(Sset, axis=0)/float(Ssize)
		self.Smean = self.beta *self.Smean + (1.0-self.beta)*self.Smean_batch
		for s in range(Ssize):
			self.row[0,:] = (Sset[s,:]-self.Smean[:])
			self.col[:,0] = (Sset[s,:]-self.Smean[:]).transpose()
			Svar[s,:,:] = self.col * self.row
		self.Sbatch = np.sum(Svar, axis=0)/float(Ssize)
			
                # Dissimilarity covariance matrix computation from the current batch
		self.Dmean_batch = np.sum(Dset, axis=0)/float(Dsize)
		self.Dmean = self.beta *self.Dmean + (1.0-self.beta)*self.Dmean_batch
		for d in range(Dsize):
			self.row[0,:] = (Dset[s,:]-self.Dmean[:])
			self.col[:,0] = (Dset[s,:]-self.Dmean[:]).transpose()
			Dvar[s,:,:] = self.col * self.row
		self.Dbatch = np.sum(Dvar, axis=0)/float(Dsize)

		# Similarity and dissimilarity covariance matrixes updating by exponentially weighted moving averages
                self.S=self.beta*self.S+(1-self.beta)*self.Sbatch
                self.D=self.beta*self.D+(1-self.beta)*self.Dbatch

		#Mahalanobis matrix computation
		self.M=np.linalg.inv(self.S+np.identity(self.S.shape[0])*0.01) - np.linalg.inv(self.D+np.identity(self.D.shape[0])*0.01)
		top[0].data[...]=self.M
		
		#SAVE MM                
                if (self.iteration%self.snapshot)==0 :
			filename = ("../WEIGHTS/MM_%i.txt" % (self.iteration))
			np.savetxt(filename, self.M)
             

	def backward(self, bottom, propagate_down, top):
		pass
	
	
