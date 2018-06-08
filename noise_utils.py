import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt
import matplotlib.pyplot as plt
from filters import FiniteDiff

try:
    from IPython import embed
except ImportError:
    pass

class NoisyState:
    def __init__(self,dt,robot=None, Ky=np.inf, Kz=np.inf):
        self.robot = robot
        self.Ky = Ky
        self.Kz = Kz
        
        self.quantum_fy = 0.0183
        self.std_fy     = 0.0134790856244
        self.std_fy     = 1.
        
        self.quantum_fz = 0.073
        self.std_fz     = 0.0814502828565
        self.std_fz     = 1.
        
        self.quantum_gyry = 0.001065
        self.std_gyry     = 0.00637104149715
        
        self.quantum_q = 8.1812e-05
        #~ self.quantum_q = 1e-03
        self.std_q     = 0.0
        
    def get_noisy_state(self,q,v,f,df):
        q_noisy = q.copy()
        v_noisy = v.copy()
        f_noisy = f.copy()
        df_noisy = df.copy()

        #Encoder noise
        q_noisy[4:]-=np.mod(q_noisy[4:],self.quantum_q)
        #Base noise
        ''''the base noise is given by the error made on the feet position due to encoder noise
        #and the noise introduced by the force sensor when evaluating the deflexion'''
        if self.robot != None:
            Mlf0, Mrf0 = self.robot.get_Mlf_Mrf(q,       update=True)  
            Mlf1, Mrf1 = self.robot.get_Mlf_Mrf(q_noisy, update=True)  
            feet_err = Mlf1.actInv(Mlf0)
            q_noisy[:2]+=feet_err.translation[1:]#y, z
            q_noisy[0] += np.matrix(np.random.normal(0,self.std_fy))/self.Ky
            q_noisy[1] += np.matrix(np.random.normal(0,self.std_fz))/self.Kz
        # here can use atan2 to get the angle, add the angular error and take sin and cos back TODO
        q_noisy[2]
        q_noisy[3]
        
        #Velocity noise
        v_noisy[0]+=np.matrix(np.random.normal(0,self.std_gyry)).T # Base velocity, equal to gyro noise assuming a 1m level arm
        
        #Force sensors Noises
        f_noisy[0]+=np.matrix(np.random.normal(0,self.std_fy)).T # Y
        f_noisy[2]+=np.matrix(np.random.normal(0,self.std_fy)).T # Y
        f_noisy[1]+=np.matrix(np.random.normal(0,self.std_fz)).T # Z
        f_noisy[3]+=np.matrix(np.random.normal(0,self.std_fz)).T # Z
        f_noisy[0]-=np.mod(f_noisy[0],self.quantum_fy) # Y
        f_noisy[2]-=np.mod(f_noisy[2],self.quantum_fy) # Y
        f_noisy[1]-=np.mod(f_noisy[1],self.quantum_fz) # Z
        f_noisy[3]-=np.mod(f_noisy[3],self.quantum_fz) # Z

        #~ return q,v,f,df
        return q_noisy,v_noisy,f_noisy,df_noisy


if __name__ == '__main__':
    dt = 1e-3
    nq = 8
    log_size = 2000
    log_q  = np.zeros([log_size,8])+np.nan 
    log_v  = np.zeros([log_size,7])+np.nan 
    log_f  = np.zeros([log_size,4])+np.nan 
    log_df = np.zeros([log_size,4])+np.nan 
    log_q_noisy  = np.zeros([log_size,8])+np.nan 
    log_v_noisy  = np.zeros([log_size,7])+np.nan 
    log_f_noisy  = np.zeros([log_size,4])+np.nan 
    log_df_noisy = np.zeros([log_size,4])+np.nan 
    
    ns = NoisyState(dt)

    for i in range(log_size):
        t = i*dt
        q=np.matrix(np.zeros(8)).T  + np.sin(t*2*np.pi*2.0)
        v=np.matrix(np.zeros(7)).T  + np.sin(t*2*np.pi*2.0)
        f=np.matrix(np.zeros(4)).T  + np.sin(t*2*np.pi*2.0)
        df=np.matrix(np.zeros(4)).T + np.sin(t*2*np.pi*2.0)
        
        q_noisy,v_noisy,f_noisy,df_noisy = ns.get_noisy_state(q,v,f,df)
        
        log_q[i] = q.A1
        log_v[i] = v.A1
        log_f[i] = f.A1
        log_df[i] = df.A1     
        log_q_noisy[i] = q_noisy.A1
        log_v_noisy[i] = v_noisy.A1
        log_f_noisy[i] = f_noisy.A1
        log_df_noisy[i] = df_noisy.A1

    plt.plot(log_q)
    plt.plot(log_q_noisy)
    plt.figure()
    plt.plot(log_f)
    plt.plot(log_f_noisy)
    plt.show()
    embed()
    
