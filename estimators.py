import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt
from IPython import embed
import matplotlib.pyplot as plt

class GenericEstimator:
    def update(self, c, dc, ddc):
        raise Exception("update method of estimator class not implemented")
        pass

class Kalman(GenericEstimator):
    def __init__(self,dt, sigma_c, sigma_dc, sigma_ddc, sigma_process_noise, c0, dc0, ddc0, dddc0):
        from LQG_utils import dkalman
        self.A = np.matrix([[1, dt, dt*dt/2, dt*dt*dt/6],
                            [0, 1,  dt,      dt*dt/2   ],
                            [0, 0,  1,       dt        ],
                            [0, 0,  0,       1         ]])
        self.B = np.matrix([0.,0.,0.,dt]).T
        self.C = np.matrix([[1.,0.,0.,0.],
                            [0.,1.,0.,0.],
                            [0.,0.,1.,0.]])
                
        self.W = sigma_process_noise**2 * np.identity(4) #Process noise
        self.V = np.diagflat([sigma_c**2 ,sigma_dc**2 ,sigma_ddc**2]) #Measument noise
        
        self.x_prev = [None,None]
        for i in [0,1]:
            self.x_prev[i] = np.matrix([[c0.A1[i]  ],
                                        [dc0.A1[i] ],
                                        [ddc0.A1[i] ],
                                        [dddc0.A1[i]]])

        (self.L,S,eigValsObs)  = dkalman(self.A, self.C, self.W, self.V)
    def update(self, c, dc, ddc, u_ddddc=np.matrix([0.,0.])):
        A = self.A
        B = self.B
        C = self.C
        L = self.L

        x_predict=[None,None]
        x_hat=[None,None]
        y=[None,None]
        u=[None,None]
        
        for i in [0,1]:
            u[i] = u_ddddc.A1[i]
            y[i] = np.matrix([[c.A1[i]  ],
                              [dc.A1[i] ],
                              [ddc.A1[i]]])            
            x_predict[i] = A * self.x_prev[i] + B * u[i]             
            x_hat[i]  = x_predict[i] + L*(y[i] - C * x_predict[i])

        #~ x_1  = A * x_0_prev + B * u + L*(y_0 - C * x_predict)
        #~ x_est  = x_predict + L*(y_1 - C * x_predict)
        c_est =   np.matrix([[x_hat[0].A1[0]],
                             [x_hat[1].A1[0]]])
        dc_est =  np.matrix([[x_hat[0].A1[1]],
                             [x_hat[1].A1[1]]])
        ddc_est = np.matrix([[x_hat[0].A1[2]],
                             [x_hat[1].A1[2]]])
        dddc_est= np.matrix([[x_hat[0].A1[3]],
                             [x_hat[1].A1[3]]])
        
        return c_est, dc_est, ddc_est, dddc_est
class FiniteDifferences(GenericEstimator):
    #This estimator simply filter c, dc, ddc
    #and compute dddc from finite differences + filtering
    def __init__(self, dt, fc_pos, fc_vel, fc_acc, fc_jerk):
        from filters import FIR1LowPass
        self.dt = dt
        self.pos_filter  = FIR1LowPass(np.exp(-2*np.pi*fc_pos*dt))
        self.vel_filter  = FIR1LowPass(np.exp(-2*np.pi*fc_vel*dt))
        self.acc_filter  = FIR1LowPass(np.exp(-2*np.pi*fc_acc*dt))
        self.jerk_filter = FIR1LowPass(np.exp(-2*np.pi*fc_jerk*dt))
        self.isFirstIter = True
    def update(self, c, dc, ddc):
        c_est = self.pos_filter.update(c)
        dc_est = self.vel_filter.update(dc)
        ddc_est = self.acc_filter.update(ddc)
        if self.isFirstIter: 
            self.ddc_prev = ddc
            self.isFirstIter = False
        dddc = (ddc - self.ddc_prev) / self.dt
        dddc_est = self.jerk_filter.update(dddc)
        self.ddc_prev = ddc
        return c_est, dc_est, ddc_est, dddc_est
        
if __name__ == '__main__':
    T = 4.0
    dt = 1e-3    
    
    def traj_sinusoid(t,start_position,stop_position,travel_time):
        # a cos(bt) + c
        A=-(stop_position-start_position)*0.5
        B = np.pi/travel_time
        C = start_position+(stop_position-start_position)*0.5
        
        p =         A*np.cos(B*t) + C
        v =      -B*A*np.sin(B*t)
        a =    -B*B*A*np.cos(B*t)
        j =   B*B*B*A*np.sin(B*t)
        s = B*B*B*B*A*np.cos(B*t) 
        return p,v,a,j,s    
    def com_traj(t):
        start_position_y,stop_position_y = 0.0 , 0.1
        start_position_z,stop_position_z = 0.53 , 0.53
        travel_time = 1.0
        py,vy,ay,jy,sy = traj_sinusoid(t,start_position_y,stop_position_y,travel_time)
        pz,vz,az,jz,sz = traj_sinusoid(t,start_position_z,stop_position_z,travel_time)
        return (np.matrix([[py ],[pz]]), 
                np.matrix([[vy ],[vz]]), 
                np.matrix([[ay ],[az]]), 
                np.matrix([[jy ],[jz]]), 
                np.matrix([[sy ],[sz]]))

    c0,dc0,ddc0,dddc0,_ = com_traj(0)
    
    # noise standard deviation
    stddev_p = 0.001 
    stddev_v = 0.01 
    stddev_a = 0.01 
    
    estimator = Kalman(dt, stddev_p,stddev_v,stddev_a, 0.01, c0,dc0,ddc0,dddc0)
    #~ estimator = FiniteDifferences(dt, 10,10,10,2)
    
    
    log_size = int(T / dt)    #max simulation samples
    log_p_gt = np.zeros([log_size,2])+np.nan    #position ground truth
    log_v_gt = np.zeros([log_size,2])+np.nan    #velocity ground truth
    log_a_gt = np.zeros([log_size,2])+np.nan    #acceleration ground truth
    log_j_gt = np.zeros([log_size,2])+np.nan    #jerk ground truth
    log_p_meas = np.zeros([log_size,2])+np.nan  #measured position
    log_v_meas = np.zeros([log_size,2])+np.nan  #measured velocity
    log_a_meas = np.zeros([log_size,2])+np.nan  #measured acceleration
    #~ log_j_meas = np.zeros([log_size,2])+np.nan  #measured jerk
    log_p_est = np.zeros([log_size,2])+np.nan   #estimated position
    log_v_est = np.zeros([log_size,2])+np.nan   #estimated velocity
    log_a_est = np.zeros([log_size,2])+np.nan   #estimated acceleration
    log_j_est = np.zeros([log_size,2])+np.nan   #estimated jerk
    

    for i in range(int(T/dt)):
        t = i*dt
        # get the ground truth data
        p_gt,v_gt,a_gt,j_gt,s_gt = com_traj(t)
        
        # add some noise to simulate sensor
        p_meas = p_gt + np.matrix(np.random.normal(0,stddev_p,2)).T
        v_meas = v_gt + np.matrix(np.random.normal(0,stddev_v,2)).T
        a_meas = a_gt + np.matrix(np.random.normal(0,stddev_a,2)).T
        
        #run the estimator
        p_est,v_est,a_est,j_est = estimator.update(p_meas,v_meas,a_meas,s_gt)
        
        #log
        log_p_gt[i] = p_gt.A1
        log_v_gt[i] = v_gt.A1
        log_a_gt[i] = a_gt.A1
        log_j_gt[i] = j_gt.A1
        log_p_meas[i] = p_meas.A1
        log_v_meas[i] = v_meas.A1
        log_a_meas[i] = a_meas.A1
        log_p_est[i] = p_est.A1
        log_v_est[i] = v_est.A1
        log_a_est[i] = a_est.A1
        log_j_est[i] = j_est.A1
        
        
        print i   
    axis = 0
    ax1 = plt.subplot(411)
    plt.plot(log_p_meas[:,axis], label="Measured Position")    
    plt.plot(log_p_est[:,axis] , label="Estimated Position")
    plt.plot(log_p_gt[:,axis]  , label="Position Ground Truth")
    plt.legend()
    ax1 = plt.subplot(412)
    plt.plot(log_v_meas[:,axis], label="Measured Velocity")    
    plt.plot(log_v_est[:,axis] , label="Estimated Velocity")
    plt.plot(log_v_gt[:,axis]  , label="Velocity Ground Truth")
    plt.legend()
    ax1 = plt.subplot(413)
    plt.plot(log_a_meas[:,axis], label="Measured Acceleration")    
    plt.plot(log_a_est[:,axis] , label="Estimated Acceleration")
    plt.plot(log_a_gt[:,axis]  , label="Acceleration Ground Truth")
    plt.legend()
    ax1 = plt.subplot(414)
    plt.plot(log_j_est[:,axis]  , label="Estimated Jerk")
    plt.plot(log_j_gt[:,axis]  , label="Jerk Ground Truth")
    plt.legend()
    plt.show()
    embed()
    
