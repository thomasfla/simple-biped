'''
Consider a Linear Dynamical System:
  dx(t)/dt = A * x(t) + a
Given x(0) and T I want to compute:
- x(T)
- The integral of x(t) for t in [0, T]
- The double integral of x(t) for t in [0, T]
'''
#from __future__ import print
import numpy as np
from numpy import matlib
from numpy.linalg import solve
from scipy.linalg import expm
from numpy.linalg import norm, eigvals
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, linewidth=200, suppress=True)

def compute_x_T(A, a, x0, T, dt=None, invertible_A=True):
    if(dt is not None):
        N = int(T/dt)
        x = matlib.copy(x0)
        for i in range(N):
            dx = A.dot(x) + a
            x += dt*dx 
        return x
        
    if(invertible_A):
        e_TA = expm(T*A)
        A_inv_a = solve(A, a)
        return e_TA*(x0+A_inv_a) - A_inv_a
    
    n = A.shape[0]
    C = matlib.zeros((n+1, n+1))
    C[0    :n,     0:n]       = A
    C[0    :n,     n]         = a
    z0 = matlib.zeros((n+1,1))
    z0[:n,0] = x0
    z0[-1,0] = 1.0
    e_TC = expm(T*C)
    z = e_TC*z0
    x_T = z[:n,0]
    return x_T
    

def compute_integral_x_T(A, a, x0, T, dt=None, invertible_A=True):
    if(dt is not None):
        N = int(T/dt)
        int_x = dt*x0
        for i in range(1, N):
            x = compute_x_T(A, a, x0, i*dt)
            int_x += dt*x 
        return int_x
        
    if(invertible_A):
        e_TA = expm(T*A)
        Ainv_a = solve(A, a)
        Ainv_x0_plus_Ainv_a = solve(A, x0+Ainv_a)
        I = matlib.eye(A.shape[0])
        return (e_TA - I)*Ainv_x0_plus_Ainv_a - T*Ainv_a
        
    n = A.shape[0]
    C = matlib.zeros((n+2, n+2))
    C[0    :n,     0:n]       = A
    C[0    :n,     n]         = a
    C[0    :n,     n+1]       = x0
    C[n    :n+1, n+1:]        = 1.0
    z0 = matlib.zeros((n+2,1))
    z0[-1,0] = 1.0
    e_TC = expm(T*C)
    z = e_TC*z0
    int_x = z[:n,0]
    return int_x

def compute_double_integral_x_T(A, a, x0, T, dt=None, compute_also_integral=False, invertible_A=True):
    if(dt is not None):
        N = int(T/dt)
        int2_x = matlib.zeros_like(x0)
        for i in range(1, N):
            int_x = compute_integral_x_T(A, a, x0, i*dt)
            int2_x += dt*int_x
        return int2_x
            
    if(invertible_A):
        e_TA = expm(T*A)
        Ainv_a = solve(A, a)
        Ainv_x0_plus_Ainv_a = solve(A, x0+Ainv_a)
        Ainv2_x0_plus_Ainv_a = solve(A, Ainv_x0_plus_Ainv_a)
        I = matlib.eye(A.shape[0])
        int2_x = (e_TA - I)*Ainv2_x0_plus_Ainv_a - T*Ainv_x0_plus_Ainv_a - 0.5*T*T*Ainv_a
        if compute_also_integral:
            int_x = (e_TA - I)*Ainv_x0_plus_Ainv_a - T*Ainv_a
            return int_x, int2_x
        return int2_x
        
    n = A.shape[0]
    C = matlib.zeros((n+3, n+3))
    C[0    :n,     0:n]       = A
    C[0    :n,     n]         = a
    C[0    :n,     n+1]       = x0
    C[n    :n+2, n+1:]        = matlib.eye(2)
    z0 = matlib.zeros((n+3,1))
    z0[-1,0] = 1.0
    e_TC = expm(T*C)
    z = e_TC*z0
    int2_x = z[:n,0]
    
    # print("A\n", A)
    # print("a\n", a.T)
    # print("x0\n", x0.T)
    # print("C\n", C)
    # print("z0\n", z0.T)
    # print("z\n", z.T)
    
    return int2_x
    
def compute_x_T_and_two_integrals(A, a, x0, T):
    '''
    Define a new variable:
      z(t) = (1, x(t), w(t), y(t))
    where:
      w(t) = int_0^t x(s) ds
      y(t) = int_0^t w(s) ds
    and then the dynamic of z(t) is:
      d1 = 0
      dx = a*1 + A*x
      dw = x
      dy = w
    which we can re-write in matrix form as:
           (0 0 0 0)
           (a A 0 0)
      dz = (0 I 0 0) z
           (0 0 I 0)
      dz =     C     z
    So we can compute x(t), w(t), y(t) by computing
      z(t) = e^{tC} z(0)
    with z(0) = (1, x(0), 0, 0)
    '''
    n = A.shape[0]
    C = matlib.zeros((3*n+1, 3*n+1))
    C[1    :1+n,   0]         = a
    C[1    :1+n,   1:1+n]     = A
    C[1+n  :1+2*n, 1:1+n]     = matlib.eye(n)
    C[1+2*n:,      1+n:1+2*n] = matlib.eye(n)
    z0 = np.vstack((1, x0, matlib.zeros((2*n,1))))
    e_TC = expm(T*C)
    z = e_TC*z0
    x      = z[1:1+n,0]
    int_x  = z[1+n:2*n+1,0]
    int2_x = z[1+2*n:,0]
    return x, int_x, int2_x

def print_error(x_exact, x_approx):
    print("Approximation error: ", np.max(np.abs(x_exact-x_approx).A1 / np.abs(x_exact).A1))
    
if __name__=='__main__':
    import time
    N_TESTS = 1000
    T = 0.001
    dt = 2e-3
    n = 16*3*2
    n2 = int(n/2)
    stiffness = 1e5
    damping = 1e2
    x0 = matlib.rand((n,1))
    a  = matlib.rand((n,1))
    U  = matlib.rand((n2, n2))
    Upsilon = U*U.T
    K = matlib.eye(n2)*stiffness
    B = matlib.eye(n2)*damping
    A = matlib.block([[matlib.zeros((n2,n2)), matlib.eye(n2)],
                      [             -Upsilon*K,      -Upsilon*B]])
    #A  = matlib.rand((n, n))

    # print("x(0) is:", x0.T)
    # print("a is:   ", a.T)
    print("State size n:", n)
    print("Eigenvalues of A:", np.sort_complex(eigvals(A)).T)
    print("")

    start_time = time.time()
    e_TA = expm(T*A)
    time_exp = time.time()-start_time
    print("Time to compute matrix exponential", 1e3*time_exp)

    start_time = time.time()
    A_inv_a = solve(A, a)
    time_solve = time.time()-start_time
    print("Time to solve linear system", 1e3*time_solve)
    print("")

    start_time = time.time()
    x_T_approx = compute_x_T(A, a, x0, T, dt)
    time_approx = time.time()-start_time

    start_time = time.time()
    x_T        = compute_x_T(A, a, x0, T)
    time_exact = time.time()-start_time
    
    start_time = time.time()
    x_T_noninv = compute_x_T(A, a, x0, T, invertible_A=False)
    time_exact_noninv = time.time()-start_time
    print("Approximated x(T) computed in             ", 1e3*time_approx)
    print("Exact x(T) computed in                    ", 1e3*time_exact)
    print("Exact x(T) for noninvertible A computed in", 1e3*time_exact_noninv)
    print_error(x_T, x_T_approx)
    print_error(x_T_noninv, x_T_approx)
    print("")

    start_time = time.time()
    int_x_T_approx = compute_integral_x_T(A, a, x0, T, dt)
    time_approx = time.time()-start_time

    start_time = time.time()
    int_x_T        = compute_integral_x_T(A, a, x0, T)
    time_exact = time.time()-start_time
    
    start_time = time.time()
    for i in range(N_TESTS): 
        int_x_T_noninv  = compute_integral_x_T(A, a, x0, T, invertible_A=False)
    time_exact_noninv = (time.time()-start_time) / N_TESTS
    print("Approximated int x(T) computed in               ", 1e3*time_approx)
    print("Exact int x(T) computed in                      ", 1e3*time_exact)
    print("Exact int x(T) for non-invertible A computed in ", 1e3*time_exact_noninv)
    print_error(int_x_T, int_x_T_approx)
    print_error(int_x_T_noninv, int_x_T_approx)
    print("")

    start_time = time.time()
    int2_x_T_approx = compute_double_integral_x_T(A, a, x0, T, dt)
    time_approx = time.time()-start_time

    start_time = time.time()
    int2_x_T        = compute_double_integral_x_T(A, a, x0, T)
    time_exact = time.time()-start_time
    
    start_time = time.time()
    for i in range(N_TESTS):
        int2_x_T_noninv = compute_double_integral_x_T(A, a, x0, T, invertible_A=False)
    time_exact_noninv = (time.time()-start_time)/N_TESTS
    
    print("Approximated int2 x(T) computed in              ", 1e3*time_approx)
    print("Exact int2 x(T) computed in                     ", 1e3*time_exact)
    print("Exact int2 x(T) for non-invertible A computed in", 1e3*time_exact_noninv)
    print_error(int2_x_T, int2_x_T_approx)
    print_error(int2_x_T_noninv, int2_x_T_approx)
    print("")
    
    start_time = time.time()    
    x_v2, int_x_v2, int2_x_v2 = compute_x_T_and_two_integrals(A, a, x0, T)
    time_exact = time.time()-start_time
    print("Test computation x(T) and first 2 integrals at once")
    print("Computation time is", 1e3*time_exact)
    print("Errors for position, integral, and double integral:")
    print_error(x_v2, x_T)
    print_error(int_x_v2, int_x_T)
    print_error(int2_x_v2, int2_x_T)