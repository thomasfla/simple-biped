import numpy as np
from simple_biped.utils.LDS_utils import compute_integrator_gains
#~ from IPython import embed
'''  
Solve inverse LQR problem for 4th order integrator
(find Q and  R leading to given gains K)  
'''
p41 = 1.1 #K1
p42 = 2.1 #K2
p43 = 3.1 #K3
p44 = 4.1 #K4

dt, p1, dp = None, -5.0, -10.0
K = compute_integrator_gains(4, p1, dp, dt)
p41, p42, p43, p44 = K[0,0], K[0,1], K[0,2], K[0,3]
#p44, p43, p42, p41 = K[0,0], K[0,1], K[0,2], K[0,3]

#~ Equations for Q to be diagonal (with R=1) !
#~ p24*p41 - p11 == 0
#~ p34*p41 - p21 == 0
#~ p31 == p41*p44
#~ p14*p42 - p11 == 0
#~ p34*p42 - p31 - p22 == 0
#~ p32 == p42*p44-p41
#~ p14*p43 - p12 == 0
#~ p24*p43 - p22 - p13 == 0
#~ -p33 == p43*p44 - p42
#~ p24*p44 - p23 - p14 == 0
#~ p34*p44 - p33 - p24 == 0
#~ p14*p44 - p13 == 0


           #   0   1   2   3   4   5   6   7   8   9  10  11
           #  p11 p12 p13 p14 p21 p22 p23 p24 p31 p32 p33 p34
M=np.matrix([[-1 , 0 , 0 , 0 , 0 , 0 , 0 ,p41, 0 , 0 , 0 , 0 ],  
             [ 0 , 0 , 0 , 0 ,-1 , 0 , 0 , 0 , 0 , 0 , 0 ,p41],   
             [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,+1 , 0 , 0 , 0 ],   
             [-1 , 0 , 0 ,p42, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],    
             [ 0 , 0 , 0 , 0 , 0 ,-1 , 0 , 0 ,-1 , 0 , 0 ,p42],    
             [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,+1 , 0 , 0 ],    
             [ 0 ,-1 , 0 ,p43, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],    
             [ 0 , 0 ,-1 , 0 , 0 ,-1 , 0 ,p43, 0 , 0 , 0 , 0 ],    
             [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 ],    
             [ 0 , 0 , 0 ,-1 , 0 , 0 ,-1 ,p44, 0 , 0 , 0 , 0 ],    
             [ 0 , 0 , 0 , 0 , 0 , 0 , 0 ,-1 , 0 , 0 ,-1 ,p44],    
             [ 0 , 0 ,-1 ,p44, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]])   
                                                                   
v=np.matrix([[     0     ],
             [     0     ],
             [  p41*p44  ],
             [     0     ],
             [     0     ],
             [p42*p44-p41],
             [     0     ],
             [     0     ],
             [p43*p44-p42],
             [     0     ],
             [     0     ],
             [     0     ]])



Minv = np.linalg.inv(M)
p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = (Minv*v).A1

#Q should be diagonal
Q =np.matrix([[       p14*p41,       p14*p42 - p11,       p14*p43 - p12,       p14*p44 - p13],
              [ p24*p41 - p11, p24*p42 - p21 - p12, p24*p43 - p22 - p13, p24*p44 - p23 - p14],
              [ p34*p41 - p21, p34*p42 - p31 - p22, p34*p43 - p32 - p23, p34*p44 - p33 - p24],
              [ p41*p44 - p31, p42*p44 - p41 - p32, p43*p44 - p42 - p33,  p44**2 - p34 - p43]])
              
np.set_printoptions(precision=5)
print "Q = "
print Q.round(10)

A=np.matrix([[0,1,0,0],
             [0,0,1,0],
             [0,0,0,1],
             [0,0,0,0]]);
            
B=np.matrix([0,0,0,1]).T

P=np.matrix([[p11,p12,p13,p14],
             [p21,p22,p23,p24],
             [p31,p32,p33,p34],
             [p41,p42,p43,p44]]);
             
print "Check ricatti's equation holds (should be zero)"
print (A.T*P + P*A - P*B*B.T*P + Q).round(10)
#~ embed()
