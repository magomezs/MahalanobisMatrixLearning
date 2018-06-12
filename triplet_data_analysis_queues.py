import caffe
import numpy as np
import os
import io
import sys

class TripletDataAnalysisQueues(caffe.Layer):
	def setup(self, bottom, top):
		#Checking inputs and outputs
		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 3 descriptors')
		if len(top)!=1:
			raise Exception('one output: Mahalanobis Matrix')
			
		#Parameters reading	
		params = eval(self.param_str)
       		self.snapshot = params["snapshot_step"]
		self.queue_size = params["queue_size"]

		self.iteration =0
                self.Squeue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Similarity queue
		self.Dqueue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Dissimilarity queue
		self.Svar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Similarity variations
		self.Dvar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Dissimilarity variations

	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')
		if bottom[0].count != bottom[2].count:
			raise Exception('Inputs must have the same dimension')
                #output has Mahalanobis Matrix dimensions
                top[0].reshape(bottom[0].channels, bottom[0].channels)

		
		
		#differnce has shape of inputs
		self.diff_p = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)    # Features difference has descriptors dimensions
		self.diff_n = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)    # Features difference has descriptors dimensions
		self.Smean = np.zeros((bottom[0].channels), dtype=np.float32)			 # Similarity expected values array
		self.Dmean = np.zeros((bottom[0].channels), dtype=np.float32)			 # Dissimilarity expedted values array
		self.S = np.identity((bottom[0].channels), dtype=np.float32)                     # Similarity covariance matrix
		self.D = np.identity((bottom[0].channels), dtype=np.float32)	                 # Dissimilarity covariance matrix
		self.M = np.identity((bottom[0].channels), dtype=np.float32)	                 # Mahalanobis matrix
		self.col= np.zeros((bottom[0].channels, 1), dtype=np.float32)
 		self.row= np.zeros((1, bottom[0].channels), dtype=np.float32)
		
	def forward(self, bottom, top):
		self.iteration=self.iteration+1
        	self.diff_p = np.absolute(bottom[0].data[...] - bottom[1].data[...])
        	self.diff_n = np.absolute(bottom[0].data[...] - bottom[2].data[...])
			
		#Move FIFO queues
		for i in range(bottom[0].num):
			for a in range(self.queue_size-1):
				self.Squeue[a,:]=self.Squeue[a+1,:]
			self.Squeue[self.queue_size-1,:] = self.diff_p[i,:] 
			for a in range(self.queue_size-1):
				self.Dqueue[a,:] = self.Dqueue[a+1,:]
			self.Dqueue[self.queue_size-1,:] =self.diff_n[i,:] 

		
	
		# Similarity covariance matrix computation
		self.Smean = np.sum(self.Squeue, axis=0)/float(self.queue_size)
		for k in range(self.queue_size):	
			self.row[0,:] = (self.Squeue[k,:]-self.Smean[:])
			self.col[:,0] = (self.Squeue[k,:]-self.Smean[:]).transpose()
			self.Svar[k,:,:] = self.col * self.row
		self.S = np.sum(self.Svar, axis=0)/float(self.queue_size)

                # Dissimilarity covariance matrix computation
		self.Dmean = np.sum(self.Dcola, axis=0)/float(self.queue_size)
		for k in range(self.queue_size):
			self.row[0,:] = (self.Dqueue[k,:]-self.Dmean[:])
			self.col[:,0] = (self.Dqueue[k,:]-self.Dmean[:]).transpose()
			self.Dvar[k,:,:] = self.col * self.row
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
          
