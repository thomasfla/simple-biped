#!/usr/bin/env python
# This class is used to log quantities evolution 
from IPython import embed
class Logger:
    def __init__(self,varNames,N=100):
        self.varNames = varNames # names of the variables we want to log
        self.i        = 0        # number of point in the logger
        self.data = {}                # dictionary containing logged data
        for varName in varNames:
            self.data[varName] = []
    def log(self,varName,value):
        
        print "in logger!"
        embed()
        self.data[varName].append(value)

def main():
    import numpy as np   
    from IPython import embed
    
    def function_that_compute_quantities(t,logger):
        cosinus = np.cos(t)
        sinus   = np.sin(t)

        # log data:
        #~ for varName in logger.varNames:
            #~ exec("logger.setDataValue(" + varName + ")=varName" )
        logger.log('sinus'  ,sinus)
        logger.log('cosinus',cosinus)
        logger.log('t',t)
        embed()
        
        
    logger = Logger(["cosinus","sinus","t"])    
    for t in np.linspace(0,2*np.pi,10):
        function_that_compute_quantities(t,logger)
     
    embed()


        
        
     
    return 0    
        
if __name__ == '__main__':
    main()

    
	

