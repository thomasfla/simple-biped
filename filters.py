''' this classes implement low pass filters that can be called iteratively with scalar or array data'''
import numpy as np
class FIR1LowPass:
    def __init__(self,alpha,name='no name'):
        '''1st order low pass FIR'''
        self.alpha = alpha
        self.name = name  
        self.isFirstData = True
    def update(self,x):
        if self.isFirstData:
            self.initfilter(x)
        self.y_prev = self.y
        self.y = x*(1-self.alpha) + self.y_prev*self.alpha
        #~ embed()
        return self.y
    def initfilter(self,x):
        self.y_prev = x;
        self.y      = x;
        self.isFirstData = False




class BALowPass:
    import numpy as np
    from scipy import signal
    
    def __init__(self,b,a,name='no name'):
        '''low pass filter given by coefficients b and a (only low pass because of init)'''
        self.b = b.copy()
        self.a = a.copy()
        self.name = name  
        self.isFirstData = True
    def update(self,x):
        if self.isFirstData:
            self.initfilter(x)
        if type(x) == np.matrix:
            x = x.copy().A1
        self.x_buff[1:,]=self.x_buff[:-1,]
        self.x_buff[0] = x

        self.y=np.dot(self.b,self.x_buff)-np.dot(self.a[1:],self.y_buff)
        
        self.y_buff[1:,]=self.y_buff[:-1,]
        self.y_buff[0] = self.y
        #~ embed()
        if self.l == 1:
            return np.asscalar(self.y)
        else:
            return self.y.reshape(self.input_shape)
        return self.y
    def initfilter(self,x):
        from IPython import embed
        self.l=1
        self.input_shape = None #
        if type(x) == np.ndarray:
            self.l=len(x)
        if type(x) == np.matrix:
            self.l=len(x)
            self.input_shape = x.shape
            x = x.copy().A1
        #~ embed()
        M = len(self.b)
        N = len(self.a)
        #~ print self.b,self.a
        #~ print "M={}".format(M)
        #~ print "N={}".format(N)
        self.x_buff = np.zeros([M,self.l])+x
        self.y_buff = np.zeros([N-1,self.l])+x
        self.y      = x;
        self.isFirstData = False
        
        
class FiniteDiff:
    def __init__(self,dt,name='no name'):
        '''finite differences'''
        self.dt = dt
        self.name = name  
        self.isFirstData = True
    def update(self,x):
        if self.isFirstData:
            self.initfilter(x)
        self.y = (x - self.x_prev) / self.dt
        self.x_prev = x
        return self.y
    def initfilter(self,x):
        self.x_prev = x;
        self.y      = x;
        self.isFirstData = False
        
        
if __name__ == "__main__":
    '''test and plot filters'''
    from IPython import embed
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    #define some silters
    filters = []
    b, a = np.array ([0.00554272,  0.01108543,  0.00554272]), np.array([1., -1.77863178,  0.80080265])
    filters.append(BALowPass(b,a,"butter_lp_filter_Wn_05_N_2"))  

    b, a = np.array ([0.09147425, -0.35947268,  0.53605377, -0.35947268,  0.09147425]), np.array([1.        , -3.7862251 ,  5.38178322, -3.40348456,  0.80798333]);   
    filters.append(BALowPass(b,a,"chebi2_lp_filter_Wn_03_N_4")) 
    
    b, a = np.array([2.16439898e-05, 4.43473520e-05, -1.74065002e-05, -8.02197247e-05,  -1.74065002e-05,   4.43473520e-05, 2.16439898e-05]),np.array([1.,-5.32595322,11.89749109,-14.26803139, 9.68705647,  -3.52968633,   0.53914042]) 
    filters.append(BALowPass(b,a,"chebi1_checby2_series")) 

    filters.append(FIR1LowPass(0.9,"1er order LP alpha = 0.9")) 
    #~ filters.append(FiniteDiff(0.001,"Finite differences")) 
   
    #~ b=b*(a.sum()/b.sum()) #make sure this is a pure low pass filter !

    #generate input data
    N=1000    
    data1d = np.zeros(N) + np.random.normal(0, 0.01, N)
    data1d[10:] += 2.0
    data1d[300:] -= 4.0
    data2d = np.zeros([N,2]) + 0.5
    data2d[20:,0] += 2.0
    data2d[5:,1] += 1.0    

    data=data2d # Test 1d or 2d data
    #allocate output memory
    data_out = []
    for filt in filters:
        data_out.append(data + np.nan)

    
    #filter data
    for i in range(N):
        for filt, out in zip(filters, data_out):
            out[i] = filt.update(data[i])
    
    #plot
    
    plt.plot(data, label ='input')
    for filt, out in zip(filters, data_out):
        plt.plot(out, label = filt.name)
    plt.legend()
    plt.show()
    embed()
