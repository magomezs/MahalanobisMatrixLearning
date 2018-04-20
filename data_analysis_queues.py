import caffe
import numpy as np
import os
import io
import sys

class DataAnalysisQueues(caffe.Layer):
	def setup(self, bottom, top):
		#Checking inputs and outputs
		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 2 descriptors and 1 label')
		if len(top)!=1:
			raise Exception('one output: MM')
			
		#Parameters reading
		params = eval(self.param_str)
       		self.snapshot = params["snapshot_step"]
		self.queue_size = params["queue_size"]
		
		self.iteration =0
                self.Squeue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Similarity queue
		self.Dqueue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Dissimilarity queue
		self.Svar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Similarity variations
		self.Dvar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Dissimilarity variations
                self.Y= np.zeros((bottom[0].num, 1), dtype=np.float32)							# Labels array


	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')
                #output has Mahalanobis Matrix dimensions
                top[0].reshape(bottom[0].channels, bottom[0].channels)

		#differnce has shape of inputs
		self.diff = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)     # Descripstors difference has descriptors dimensions
                self.Smean = np.zeros((bottom[0].channels), dtype=np.float32)			# Similarity expected values array
		self.Dmean = np.zeros((bottom[0].channels), dtype=np.float32)			# Dissimilarity expedted values array
		self.S = np.identity((bottom[0].channels), dtype=np.float32)			# Similarity covariance matrix
		self.D = np.identity((bottom[0].channels), dtype=np.float32)			# Dissimilarity covariance matrix
		self.M = np.identity((bottom[0].channels), dtype=np.float32)			# Mahalanobis matrix
		 	
       		
	def forward(self, bottom, top):
		self.iteration=self.iteration+1
		self.Y[:,0] = bottom[2].data[...]
        	self.diff = bottom[0].data[...] - bottom[1].data[...]
	
		#Move FIFO queues
		for i in range(bottom[0].num):
			if self.Y[i] == 1.0 :
				for a in range(self.queue_size-1):
					self.Squeue[a,:]=self.Squeue[a+1,:]
				self.Squeue[self.queue_size-1,:] = self.diff[i,:] 	
			else:
				for a in range(self.queue_size-1):
					self.Dqueue[a,:] = self.Dqueue[a+1,:]
				self.Dqueue[self.queue_size-1,:] =self.diff[i,:] 
				
		# Similarity covariance matrix computation
		self.Smean = np.sum(self.Squeue, axis=0)/float(self.queue_size)
		for k in range(self.queue_size):
			self.Svar[k,:,:] = (self.Squeue[k,:]-self.Smean[:]).transpose() * (self.Squeue[k,:]-self.Smean[:])
		self.S = np.sum(self.Svar, axis=0)/float(self.queue_size)

		# Dissimilarity covariance matrix computation
		self.Dmean = np.sum(self.Dqueue, axis=0)/float(self.queue_size)
		for k in range(self.queue_size):
			self.Dvar[k,:,:] = (self.Dqueue[k,:]-self.Dmean[:]).transpose() * (self.Dqueue[k,:]-self.Dmean[:])
		self.D = np.sum(self.Dvar, axis=0)/float(self.queue_size)

		#Mahalanobis matrix computation
		self.M=np.linalg.inv(self.S+np.identity(self.S.shape[0])*0.01) - np.linalg.inv(self.D+np.identity(self.D.shape[0])*0.01)
		top[0].data[...]=self.M
		
		#SAVE MM
                if (self.iteration%self.snapshot)==0 :
			filename = ("../WEIGHTS/MM_%i.txt" % (self.iteration))
			np.savetxt(filename, self.M)
             

	def backward(self, bottom, propagate_down, top):
		pass
