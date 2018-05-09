from scipy import linalg as la
def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151 
    #first, try to solve the ricatti equation
    X = la.solve_continuous_are(A, B, Q, R)
    #compute the LQR gain
    K = la.solve(R, B.T.dot(X))
    eigVals, eigVecs = la.eig(A-B.dot(K))
    return K, X, eigVals
 
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.     
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151 
    #first, try to solve the ricatti equation
    X = la.solve_discrete_are(A, B, Q, R)
    #compute the LQR gain
    K = la.solve(B.T.dot(X).dot(B)+R, B.T.dot(X).dot(A))    
    eigVals, eigVecs = la.eig(A-B.dot(K))     
    return K, X, eigVals
    
def dkalman(A,C,W,V):
    """Solve the infinite-horizon discrete-time Kalman observer.     
     
    x[k+1] = A x[k] + B u[k] + w[y]
    y[k]   = C x[k] + v[t]
     
    """
    #first, try to solve the ricatti equation
    S = la.solve_discrete_are(A.T, C.T, W, V)
    #compute the Kalman gain
    L = la.solve(C.dot(S).dot(C.T)+V, C.dot(S).dot(A.T)).T
    eigVals, eigVecs = la.eig(A-L.dot(C))
    return L, S, eigVals
