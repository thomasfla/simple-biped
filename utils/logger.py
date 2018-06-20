#!/usr/bin/env python
# This class is used to log quantities evolution 
import numpy as np   
from RAI.data_collector import DataCollector

try:
    from IPython import embed
except ImportError:
    pass

class Empty:
    def __init__(self):
        pass

class SimpleArrayLogger:
    ''' Logger utility class for numpy.array data.
    '''
    def __init__(self, N):
        self.N      = N         # max number of times each variable will be logged
        self.i      = 0         # number of times variables have been logged
        self.obj_list = []
        self.obj_name_list = []
        self.var_names_list = []
            
    def add_variables(self, obj, var_names, obj_name=None, log_names=None):
        self.obj_list.append(obj)
        self.obj_name_list.append(obj_name)
        self.var_names_list.append(var_names)
    
    def log_all(self):
        data = self.__dict__
        i = self.i
        
        for (obj, obj_name, var_names) in zip(self.obj_list, self.obj_name_list, self.var_names_list):
            for local_var_name in var_names:
                if(obj_name is None):
                    var_name = local_var_name
                else:
                    var_name = obj_name+'_'+local_var_name
#                print "Time %d Variable %s, value"%(i, var_name), obj.__dict__[local_var_name]
                
                # check if the memory has already been allocated for this variable
                if(var_name not in data.keys()):
                    data[var_name] = np.empty((self.N, obj.__dict__[local_var_name].size))
                    
                data[var_name][i,:] = np.asarray(obj.__dict__[local_var_name])
        self.i += 1

class RaiLogger(DataCollector):
    
    def __init__(self):
        super(RaiLogger,self).__init__()
        self.i      = 0         # number of times the method log_all has been called
        self.obj_list = []
        self.log_var_name_list = []     # names used in logger
        self.obj_var_names_list = []    # names used in original object
        self.var_types_list = []
        
    def auto_log_variables(self, obj, var_names, var_types, obj_name=None, log_var_names=None):
        ''' 
        Add the given variables to the list of variables to automatically log every time 
        the method log_all() is called.
        
        :param obj: object containing the variables to log
        :param var_names: list of names of the variables to log (names used on the object)
        :param var_types: list of data types (which can be either 'variable', 'vector', 'vector3d', 'quaternion', 'rpy', 'se3', or 'matrix')
        :param obj_name: if specified, the variable names used for logging begin with: <obj_name>_<var_name>
        :param log_var_names: the variable names used for logging (optional)
        '''
        self.obj_list.append(obj)
        self.obj_var_names_list.append(var_names)
        self.var_types_list.append(var_types)

        if(log_var_names is not None):
            self.log_var_name_list.append(log_var_names)
            return
            
        # compute variable names used for logging
        log_var_names = []
        for (obj_var_name, var_type) in zip(var_names, var_types):
            if(obj_name is None):
                var_name = obj_var_name
            else:
                var_name = obj_name+'_'+obj_var_name
                
            if(var_type=='variable'):
                var_names = [var_name]
            elif(var_type=='vector'):
                try:
                    # try to access the variable to see how many elements are in the vector
                    var = obj.__dict__[obj_var_name]
                    var_names = [var_name+'_'+str(i) for i in range(self._vector_size(var))]
                except:
#                    print "[RaiLogger] Failed to access variable", var_name, "to count vector items"
                    var_names = [var_name,]
            elif(var_type=='vector3d'):
                var_names = [var_name+'_'+coord for coord in ['x', 'y', 'z']]
            elif(var_type=='quaternion'):
                var_names = [var_name+'_'+coord for coord in ['qx', 'qy', 'qz', 'qw']]
            elif(var_type=='rpy'):
                var_names = [var_name+'_'+coord for coord in ['roll', 'pitch', 'yaw']]
            elif(var_type=='se3'):
                var_names = [var_name+'_'+coord for coord in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
            elif(var_type=='matrix'):
                var_names = [var_name+'_'+str(row)+'_'+str(col) for row in range(var.shape[0]) for col in range(var.shape[1])]
            else:
                raise ValueError('Unknown data type '+var_type+' for variable '+obj_var_name)
            
            log_var_names.append(var_names)
        
        self.log_var_name_list.append(log_var_names)
        

    def auto_log_local_variables(self, var_names, var_types, prefix=None, log_var_names=None):
        ''' 
        Add the given local variables to the list of variables to automatically log every time 
        the method log_all is called. Note that for this to work, a pointer to the locals() 
        dictionary should be passed to the log_all method.
        
        :param var_names: list of names of the local variables to log
        :param var_types: list of data types (which can be either 'variable', 'vector', 'vector3d', 'quaternion', 'rpy', 'se3', or 'matrix')
        :param prefix: if specified, the variable names used for logging begin with: <prefix>_<var_name>
        :param log_var_names: the variable names used for logging (optional)
        '''
        self.auto_log_variables(None, var_names, var_types, obj_name=prefix, log_var_names=log_var_names)
    
    
    def log_all(self, local_dict=None):
        # at the first iteration update the name of the variables used for logging
        # based on the size of the associated vectors
        if(self.i==0):
            # create a new list to avoid modifying the list while iterating over it
            new_log_var_name_list = []
            
            for (obj, obj_var_names, log_var_names, var_types) in zip(self.obj_list, self.obj_var_names_list, self.log_var_name_list, self.var_types_list):
                new_log_var_names = []
                for (obj_var_name, log_var_name, var_type) in zip(obj_var_names, log_var_names, var_types):
                    try:
                        if(obj is None):
                            var = local_dict[obj_var_name]
                        else:
                            var = obj.__dict__[obj_var_name]
                    except KeyError:
                        print "Could not find field %s of type %s" % (obj_var_name, var_type)
                        raise
                    
                    # by default assume the log-var-name will not change
                    new_log_var_name = log_var_name
                    
                    if(var_type=='vector' and len(log_var_name) != var.size):
                        # in case it was not possible to access the variable when auto_log_variables was called
                        vn = log_var_name[0]
                        new_log_var_name = [vn+'_'+str(i) for i in range(self._vector_size(var))]
#                        print "log_var_name for vector variable %s (obj_var_name=%s) is:"%(vn, obj_var_name), new_log_var_name                    
                    new_log_var_names.append(new_log_var_name)
                    
                new_log_var_name_list.append(new_log_var_names)
                
            self.log_var_name_list = new_log_var_name_list
            
        for (obj, obj_var_names, log_var_names, var_types) in zip(self.obj_list, self.obj_var_names_list, self.log_var_name_list, self.var_types_list):
            for (obj_var_name, log_var_name, var_type) in zip(obj_var_names, log_var_names, var_types):
                if(obj is None):
                    var = local_dict[obj_var_name]
                else:
                    var = obj.__dict__[obj_var_name]
                    
                if(var_type=='variable'):
                    self.add_variable(var, log_var_name[0], unit="")
                elif(var_type=='vector'):
                    self.add_vector(var, log_var_name)
                elif(var_type=='vector3d'):
                    self.add_vector(var, log_var_name)
                elif(var_type=='quaternion'):
                    self.add_vector(var, log_var_name)
                elif(var_type=='rpy'):
                    self.add_vector(var, log_var_name)
                elif(var_type=='se3'):
                    self.add_se3(var, log_var_name[0][:-4], unit="")
                elif(var_type=='matrix'):
                    self.add_matrix(var, log_var_name[0][:-4], unit="")
                else:
                    raise ValueError('Unknown data type '+var_type+' for variable '+obj_var_name)
                
                # store link to data as member variable for easy access
                for vn in log_var_name:
                    if(vn not in self.__dict__.keys()):
                        self.__dict__[vn] = self.data[self.fields[vn]]
                    
        self.i += 1
        
    def add_vector(self, data_vec, field, unit_vec=None):
        """
        Slight improvement over the same method of the parent class which allows
        the user to specify a single string for the field and nothing for the unit
        """
        
        if isinstance(field, basestring):
            field_vec = [field+'_'+str(i) for i in range(self._vector_size(data_vec))]
        else:
            field_vec = field
        
        if(unit_vec is None):
            unit_vec = self._vector_size(data_vec)*['',]
            
        try:
            super(RaiLogger,self).add_vector(data_vec, field_vec, unit_vec)
        except:
            print "Error while adding vector %s with data:"%(field), data_vec
            raise

def exampleSimpleArrayLogger():    
    N = 10
    logger = SimpleArrayLogger(N)

    obj1 = Empty()
    obj1.a = np.array([3, 4])
    obj1.b = np.array([18.0])
    
    obj2 = Empty()
    obj2.a = np.array([4, 5])
    obj2.b = np.array([19.0])
    
    logger.add_variables(obj1, ['a', 'b'], 'obj1')
    logger.add_variables(obj2, ['a', 'b'], 'obj2')
    
    for t in np.linspace(0,2*np.pi,N):
        logger.log_all()
        obj1.a = np.array([t, t+1])
        obj1.b = np.array([t**2])
        obj2.a = obj1.a*3
        obj2.b = obj1.b*3
    
    return logger
    
    
def exampleRaiLogger():
    N = 10
    logger = RaiLogger()

    obj1 = Empty()
    obj1.a = np.array([3, 4])
    obj1.b = np.array([18.0])
    obj1.c = np.matrix([0.1, 0.2, 0.3])
    
    obj2 = Empty()
    
    logger.auto_log_variables(obj1, ['a', 'b', 'c'], ['vector', 'variable', 'vector3d'], 'obj1')
    logger.auto_log_variables(obj2, ['a', 'b'], ['vector', 'variable'], 'obj2')
    
    obj2.a = np.array([4, 5])
    obj2.b = np.array([19.0])
    
    for t in np.linspace(0,2*np.pi,N):
        logger.log_all()
        obj1.a = np.array([t, t+1])
        obj1.b = np.array([t**2])
        obj1.c = obj1.c*2
        obj2.a = obj1.a*3
        obj2.b = obj1.b*3
    
    return logger  
    
    
def test_logging_time():
    ''' Compare the computation time taking for logging using two different strategiiies:
        - memory pre-allocation
        - list append
    '''
    import time
    
    N = int(1e6)    # number of data points to log
    m = 10          # size of one data point
    data = np.random.random((N,m))
    
    data_log = np.empty((N,m));    
    start = time.time()
    for i in range(N):
        data_log[i,:] = data[i,:]
    end = time.time()
    print "Time take to log %d data points of size %d with memory pre-allocation:         "%(N,m), end - start
    
    data_log = []
    start = time.time()
    for i in range(N):
        data_log.append(data[i,:])
    end = time.time()
    print "Time take to log %d data points of size %d with list append of vectors:        "%(N,m), end - start
    
    data_log = []
    start = time.time()
    for i in range(N):
        for j in range(m):
            data_log.append(data[i,j])
    end = time.time()
    print "Time take to log %d data points of size %d with list append of scalar values:  "%(N,m), end - start
    
    data = np.random.random((m,N))
    
    data_log = np.empty((m,N));    
    start = time.time()
    for i in range(N):
        data_log[:,i] = data[:,i]
    end = time.time()
    print "Time take to log %d data points of size %d with col-wise memory pre-allocation:"%(N,m), end - start
    
    
        
if __name__ == '__main__':
    logger = exampleSimpleArrayLogger()
#    logger = exampleRaiLogger()
#    test_logging_time()
