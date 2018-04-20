import caffe
import numpy as np
import os
import sys


class ConnectionFunction(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 2 features arrays and 1 MM (Mahalanobis Matrix)')
		if len(top)!=1:
			raise Exception('must have exactrly one output: distance = connection function')
		self.iteration =0
		
		#LOAD Mahalanobis matrix
		self.M = np.identity((bottom[0].channels), dtype=np.float32)
		
		#Parameter reading
		param = eval(self.param_str)
       		self.changing_distance_it = param["changing_distance_it"]
          

	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')
                if bottom[2].shape[0] != bottom[2].shape[1]:
                        raise Exception('MM must be squared')
                if bottom[2].shape[0] != bottom[0].channels:
                        raise Exception('MM dimension must be descriptors dimension')

		
		self.diff = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)  	#Descripstors difference has descriptors dimensions
       		self.dist = np.zeros((bottom[0].num, 1), dtype=np.float32)			#Connection function
		
        	#connection function = output 
		top[0].reshape(bottom[0].num)

		
	def forward(self, bottom, top):
		self.iteration=self.iteration + 1
		self.diff = bottom[0].data - bottom[1].data
		
		if(self.iteration < self.changing_distance_it):	
        		#Euclidean distance computation
			self.dist[..., 0] = np.sqrt(np.sum(self.diff**2, axis=1))   
			top[0].data[...]=self.dist[..., 0]
	
		else:		
			#Mahalanobis distance computation						
			self.M = bottom[2].data[...]					#read mahalanobis matrix
			for i in range(bottom[0].num):
                       		A=np.zeros(bottom[0].channels, dtype=np.float32)
				for j in range (bottom[0].channels):
                        		A[j]=np.sum(self.diff[i,:]*self.M[:,j])
                        	top[0].data[i]=sum(A*self.diff[i,:])

			
	def backward(self, bottom, propagate_down, top):
		pass

