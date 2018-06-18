import numpy as np

def restert_viewer_server(procName='gepetto-gui', delay = 0.2):
    ''' This function kill and restart the viewer server'''
    import os, signal, subprocess
    from time import sleep
    def check_kill_process(pstring):
        for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            os.kill(int(pid), signal.SIGKILL)
        check_kill_process(procName)
    sleep(delay)
    proc = subprocess.Popen([procName], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    sleep(delay)


def finite_diff(data, dt):
    fd = np.asarray(data).copy()
    fd[0] = 0.
    fd[1:,] -= np.asarray(data[:-1])
    fd = fd *(1/dt)
    fd[0] = np.nan # just ignore the first points for display
    return fd
    

def traj_sinusoid(t, start_position, stop_position, travel_time):
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
    
# check derivatives
#eps = 1e-8
#p1,v1,a1,j1,s1 =  traj_sinusoid(0.123   ,1.1,2.2,3.3)
#p2,v2,a2,j2,s2 =  traj_sinusoid(0.123+eps,1.1,2.2,3.3)
#assert isapprox(v1,(p2-p1)/eps)
#assert isapprox(a1,(v2-v1)/eps)
#assert isapprox(j1,(a2-a1)/eps)
#assert isapprox(s1,(j2-j1)/eps)