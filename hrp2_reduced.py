import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt

try:
    from IPython import embed
except ImportError:
    pass

class Visual:
    '''
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).
    '''
    def __init__(self,name,jointParent,placement):
        self.name = name                  # Name in gepetto viewer
        self.jointParent = jointParent    # ID (int) of the joint 
        self.placement = placement        # placement of the body wrt joint, i.e. bodyMjoint
    def place(self,display,oMjoint,refresh=True):
        oMbody = oMjoint*self.placement
        display.applyConfiguration(self.name,
                                   se3ToXYZQUAT(oMbody) )
        #Hotfix, my viewer wants the older order
        #~ x,y,z,a,b,c,d = se3ToXYZQUAT(oMbody)
        #~ display.applyConfiguration(self.name,
                                   #~ [x,y,z,d,a,b,c] )
        if refresh: display.refresh()

class Hrp2Reduced:
    '''
    Class following robot-wrapper template, with a model, a data and a viewer object.
    Init from the urdf model of HRP-2.
    Display with robot.display. Initial configuration set to self.q0.
    '''
    def __init__(self,urdf,pkgs,loadModel=True,useViewer=True):
        '''Typical call with Hrp2Reduced('..../hrp2.udf',['dir/with/mesh/files']).'''
        self.useViewer = useViewer
        self.loadHRP(urdf,pkgs,loadModel)
        self.buildModel()
        
    def loadHRP(self,urdf,pkgs,loadModel):
        '''Internal: load HRP-2 model from URDF.'''
        robot = RobotWrapper( urdf, pkgs, root_joint=se3.JointModelFreeFlyer() )
        useViewer = self.useViewer
        if useViewer:
            robot.initDisplay( loadModel = loadModel)
            if 'viewer' not in robot.__dict__: robot.initDisplay()
            for n in robot.visual_model.geometryObjects: robot.viewer.gui.setVisibility(robot.viewerNodeNames(n),'ON')

        robot.q0 = np.matrix( [
                0., 0., 0.648702, 0., 0. , 0., 1.,               # Free flyer
                0., 0., 0., 0.,                                  # Chest and head
                0.261799, 0.17453,  0., -0.523599, 0., 0., 0.1,  # Left arm
                0.261799, -0.17453, 0., -0.523599, 0., 0., 0.1,  # Right arm
                0., 0., -0.453786, 0.872665, -0.418879, 0.,      # Left leg
                0., 0., -0.453786, 0.872665, -0.418879, 0.,      # Righ leg
                ] ).T

        self.hrpfull = robot

    def buildModel(self):
        '''Internal:build a 3+2+2 DOF model representing a simplified version of HRP-2 legs.'''
        robot = self.hrpfull
        se3.crba(robot.model,robot.data,robot.q0)
        se3.forwardKinematics(robot.model,robot.data,robot.q0)
        WAIST_ID = 1
        CHEST_ID = 2
        RHIP_ID  = 26
        LHIP_ID  = 20
        RTHIGH_ID = 28
        LTHIGH_ID = 22
        RTIBIA_ID = 29
        LTIBIA_ID = 23
        RKNEE_ID = 29
        LKNEE_ID = 23
        RANKLE_ID = 31
        LANKLE_ID = 25

        if self.useViewer:
            self.viewer  = viewer   = robot.viewer.gui
        self.model   = modelred = se3.Model.BuildEmptyModel()
        self.visuals = visuals  = []

        colorwhite   = [red,green,blue,transparency] = [1,1,0.78,1.0]
        colorred     = [1.0,0.0,0.0,1.0]
        colorblue    = [0.0,0.0,0.8,1.0]

        jointId = 0

        # Add first (free-floating) joint
        name          = "waist"
        JointModel    = se3.JointModelPlanar
        inertia       = (robot.data.oMi[WAIST_ID].inverse()*robot.data.oMi[CHEST_ID])\
            .act(robot.data.Ycrb[CHEST_ID])+robot.model.inertias[WAIST_ID]
        placement     = se3.SE3(rotate('x',pi/2)*rotate('y',pi/2),zero(3))
        t = placement.translation;        t[0] = 0;        placement.translation=t

        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = modelred.addJoint(jointId,JointModel(),placement,jointName)
        modelred.appendBodyToJoint(jointId,inertia,se3.SE3.Identity())
        if self.useViewer:
            viewer.addSphere('world/red'+bodyName, 0.05,colorwhite)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add right hip joint
        name = "rhip"
        JointModel = se3.JointModelRX
        inertia    = robot.model.inertias[RTHIGH_ID]
        placement = robot.data.oMi[WAIST_ID].inverse()*robot.data.oMi[RHIP_ID]
        t = placement.translation;        t[0] = 0;   t[1] = -0.095;     placement.translation=t
        placement  = se3.SE3(rotate('x',-pi/2)*rotate('z',-pi/2),zero(3))*placement

        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = modelred.addJoint(jointId,JointModel(),placement,jointName)
        modelred.appendBodyToJoint(jointId,inertia,se3.SE3.Identity())
        if self.useViewer:
            viewer.addSphere('world/red'+bodyName, 0.05,colorred)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add right knee joint
        name = "rknee"
        JointModel = se3.JointModelPZ
        inertia = robot.model.inertias[RTIBIA_ID]
        placement = robot.data.oMi[RHIP_ID].inverse()*robot.data.oMi[RKNEE_ID]
        t = placement.translation;        t[0] = 0;        placement.translation=t
        placement.rotation = rotate('y',0)

        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = modelred.addJoint(jointId,JointModel(),placement,jointName)
        modelred.appendBodyToJoint(jointId,inertia,se3.SE3.Identity())
        #viewer.addSphere('world/red'+bodyName, 0.05,colorred)
        if self.useViewer:
            viewer.addBox('world/red'+bodyName, 0.05,0.05,0.05,colorred)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add right ankle spot
        name = "rankle"
        placement = robot.data.oMi[RKNEE_ID].inverse()*robot.data.oMi[RANKLE_ID]
        t = placement.translation;        t[0] = 0;   t[1]=0;     placement.translation=t
        placement.rotation = rotate('y',0)

        modelred.addFrame(se3.Frame(name,jointId,0,placement,se3.FrameType.OP_FRAME))
        if self.useViewer:
            viewer.addSphere('world/red'+name, 0.05,colorred)
        visuals.append( Visual('world/red'+name,jointId,placement*se3.SE3(eye(3),np.matrix([0,0,.05]).T)) )
        if self.useViewer:
            viewer.addBox('world/red'+name+'sole', 0.15,.06,.01,colorred)
        visuals.append( Visual('world/red'+name+'sole',jointId,placement) )

        # Add left hip
        name = "lhip"
        JointModel = se3.JointModelRX
        inertia    = robot.model.inertias[LTHIGH_ID]
        placement  = robot.data.oMi[WAIST_ID].inverse()*robot.data.oMi[LHIP_ID]
        t = placement.translation;        t[0] = 0;   t[1]=0.095;     placement.translation=t
        placement  = se3.SE3(rotate('x',-pi/2)*rotate('z',-pi/2),zero(3))*placement

        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = modelred.addJoint(1,JointModel(),placement,jointName)
        modelred.appendBodyToJoint(jointId,inertia,se3.SE3.Identity())
        if self.useViewer:
            viewer.addSphere('world/red'+bodyName, 0.05,colorblue)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add left knee
        name = "lknee"
        JointModel = se3.JointModelPZ
        inertia = robot.model.inertias[LTIBIA_ID]
        placement = robot.data.oMi[LHIP_ID].inverse()*robot.data.oMi[LKNEE_ID]
        t = placement.translation;        t[0] = 0;        placement.translation=t
        placement.rotation = rotate('y',0)

        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = modelred.addJoint(jointId,JointModel(),placement,jointName)
        modelred.appendBodyToJoint(jointId,inertia,se3.SE3.Identity())
        #viewer.addSphere('world/red'+bodyName, 0.05,colorblue)
        if self.useViewer:
            viewer.addBox('world/red'+bodyName, 0.05,0.05,0.05,colorblue)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add left ankle spot
        name = "lankle"
        placement = robot.data.oMi[LKNEE_ID].inverse()*robot.data.oMi[LANKLE_ID]
        t = placement.translation;        t[0] = 0;   t[1]=0;     placement.translation=t
        placement.rotation = rotate('y',0)

        modelred.addFrame(se3.Frame(name,jointId,0,placement,se3.FrameType.OP_FRAME))
        if self.useViewer:
            viewer.addSphere('world/red'+name, 0.05,colorblue)
        visuals.append( Visual('world/red'+name,jointId,placement*se3.SE3(eye(3),np.matrix([0,0,.05]).T)) )
        if self.useViewer:
            viewer.addBox('world/red'+name+'sole', 0.15,.06,.01,colorblue)
        visuals.append( Visual('world/red'+name+'sole',jointId,placement) )

        self.data = modelred.createData()
        self.q0   =  np.matrix([0,0.569638,1.,0,0,0,0,0]).T
        self.RF = self.model.getFrameId('rankle')
        self.LF = self.model.getFrameId('lankle')
        self.RK = self.model.frames[self.RF].parent
        self.LK = self.model.frames[self.LF].parent
        
    def get_Jl_Jr_world(self, q, update=True):
        if update==True:
            se3.forwardKinematics(self.model,self.data,q)
            se3.computeJacobians(self.model,self.data,q)
            se3.framesKinematics(self.model,self.data,q)
        oMlf, oMrf = self.get_Mlf_Mrf(q,False)
        rotLF = se3.SE3(oMlf.rotation,0*oMlf.translation)
        rotRF = se3.SE3(oMrf.rotation,0*oMrf.translation)
        Jl = se3.frameJacobian(self.model,self.data,self.LF)
        Jr = se3.frameJacobian(self.model,self.data,self.RF)        
        Jl2 = rotLF.action * Jl
        Jr2 = rotRF.action * Jr
        return Jl2,Jr2 #return the rotated jacobian (WORLD)
        #~ return Jl,Jr # (LOCAL)
        
    def get_Mlf_Mrf(self, q, update=True):
        if update==True:
            se3.forwardKinematics(self.model,self.data,q)
            se3.framesKinematics(self.model,self.data,q)
        oMlf = self.data.oMf[self.LF].copy()
        oMrf = self.data.oMf[self.RF].copy()
        return oMlf, oMrf
        
    def get_classic_alf_arf(self,q,v,a,update=True):
        if update==True:
            se3.forwardKinematics(self.model,self.data,q,v,a)
            se3.framesKinematics(self.model,self.data,q)        
        v_LF,v_RF = self.get_vlf_vrf_world(q,v,False)
        a_LF,a_RF = self.get_alf_arf_world(q,v,a,False)
        classic_alf = a_LF
        classic_arf = a_RF
        classic_alf.linear += np.cross(v_LF.angular.T, v_LF.linear.T).T
        classic_arf.linear += np.cross(v_RF.angular.T, v_RF.linear.T).T
        return classic_alf,classic_arf
        
    def get_driftLF_driftRF_world(self, q, v, update=True):
        '''return dJdq of the feet in the world frame'''
        if update==True:
            se3.forwardKinematics(self.model,self.data,q,v,0*v)
            se3.framesKinematics(self.model,self.data,q)
        driftLF,driftRF = self.get_classic_alf_arf(q,v,0*v,False)
        #~ v_LF,v_RF = self.get_vlf_vrf_world(q,v,False)
        #~ a_LF,a_RF = self.get_alf_arf_world(q,v,0*v,False)
        #~ driftLF = a_LF
        #~ driftRF = a_RF
        #~ driftLF.linear += np.cross(v_LF.angular.T, v_LF.linear.T).T
        #~ driftRF.linear += np.cross(v_RF.angular.T, v_RF.linear.T).T
        return driftLF,driftRF
        
    def get_vlf_vrf_world(self, q, v, update=True):
        if update==True:
            se3.forwardKinematics(self.model,self.data,q,v)
            se3.framesKinematics(self.model,self.data,q)
        Mlf,Mrf = self.get_Mlf_Mrf(q, False)
        Rlf = se3.SE3(Mlf.rotation, 0*Mlf.translation)
        Rrf = se3.SE3(Mrf.rotation, 0*Mrf.translation)
        vlf = self.model.frames[self.LF].placement.inverse()*self.data.v[self.LK]
        vrf = self.model.frames[self.RF].placement.inverse()*self.data.v[self.RK]
        vlf = Rlf.act(vlf)
        vrf = Rrf.act(vrf)
        return vlf,vrf    

    def get_alf_arf_world(self, q, v, a, update=True):
        if update==True:
            se3.forwardKinematics(self.model,self.data,q,v,a)
            se3.framesKinematics(self.model,self.data,q)
        Mlf,Mrf = self.get_Mlf_Mrf(q,False)
        Rlf = se3.SE3(Mlf.rotation, 0*Mlf.translation)
        Rrf = se3.SE3(Mrf.rotation, 0*Mrf.translation)
        alf = self.model.frames[self.LF].placement.inverse()*self.data.a[self.LK]
        arf = self.model.frames[self.RF].placement.inverse()*self.data.a[self.RK]
        alf = Rlf.act(alf)
        arf = Rrf.act(arf)
        return alf,arf   

    def get_angularMomentumJacobian(self, q, v, update=True):
        if(update):
            se3.ccrba(self.model, self.data, q, v);
        Jam = self.data.Ag[3:]
        return Jam;
    
    def get_angularMomentum(self, q, v, update=True):
        Jam = self.get_angularMomentumJacobian(q, v, update)
        am = (Jam*v).A1[0]
        return am
        
    def get_angular_momentum_and_derivatives(self, q, v, f=None, df=None, Ky=None, Kz=None, recompute=True):
        Mlf, Mrf = self.get_Mlf_Mrf(q, recompute)
        com_state = self.get_com_and_derivatives(q, v, f, df, recompute)
        Jam = self.get_angularMomentumJacobian(q,v)  
        
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        cy, cz     = com_state[0].A1
        dcy, dcz   = com_state[1].A1
        am = (Jam*v).A1[0]
#        robotInertia = Jam[0,2] 
#        theta = np.arctan2(q[3],q[2])
#        iam   = robotInertia * theta
        if(f is None):
            return am
        
        fyl, fzl, fyr, fzr = f.A1
        ddcy, ddcz = com_state[2].A1  
        dam   = (pyl-cy)*fzl                  -  (pzl-cz)*fyl                  + (pyr-cy)*fzr                 -  (pzr-cz)*fyr
        if(df is None):
            return a, dam
            
        dfyl, dfzl, dfyr, dfzr = df.A1
        dpyl, dpzl, dpyr, dpzr = -dfyl/Ky, -dfzl/Kz, -dfyr/Ky, -dfzr/Kz
        ddam  = (dpyl-dcy)*fzl+(pyl-cy)*dfzl - ((dpzl-dcz)*fyl+(pzl-cz)*dfyl) + (dpyr-dcy)*fzr+(pyr-cy)*dfzr - ((dpzr-dcz)*fyr+(pzr-cz)*dfyr)
        return am, dam, ddam
        
    def get_com_and_derivatives(self, q, v, f=None, df=None, recompute=True):
        '''Compute the CoM position, velocity, acceleration and jerk from 
           q, v, contact forces and deritavives of the contact forces'''
        if recompute:
            se3.centerOfMass(self.model,self.data,q,v,zero(self.model.nv))
        com   = self.data.com[0][1:]
        com_v = self.data.vcom[0][1:]
        if(f is None):
            return com, com_v
            
        m = self.data.mass[0]
        X = np.hstack([np.eye(2),np.eye(2)])
        com_a = (1/m)*X*f + self.model.gravity.linear[1:]
        if(df is None):
            return com, com_v, com_a    
            
        com_j = (1/m)*X*df
        return com, com_v, com_a, com_j
        
    def compute_torques_from_dv_and_forces(self, dv, f):
        M  = self.data.M        #(7,7)
        h  = self.data.nle      #(7,1)
        Jl,Jr = self.get_Jl_Jr_world(q, False)
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        tau = (M*dv - Jc.T*f + h)[3:]
        return tau
        
    def display(self,q):
        se3.forwardKinematics(self.model,self.data,q)
        if self.useViewer:
            for visual in self.visuals:
                visual.place( self.viewer,self.data.oMi[visual.jointParent] )
                self.viewer.refresh()

if __name__ == '__main__':
    import time
    from path import pkg, urdf 
    useViewer = False
    if useViewer: 
        from utils.utils_thomas import restert_viewer_server
        restert_viewer_server()
    
    robot = Hrp2Reduced(urdf,[pkg],useViewer=useViewer)
    robot.display(robot.q0)
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=200)
    RF = robot.model.getFrameId('rankle')
    LF = robot.model.getFrameId('lankle')
    RK = robot.model.frames[RF].parent
    LK = robot.model.frames[LF].parent
    q = robot.q0.copy()
    v = np.zeros([7,1])
    teta = 0.5
    q = np.matrix([[ 0.1            ],
                    [ 0.569638      ],
                    [ np.cos(teta)  ],
                    [ np.sin(teta)  ],
                    [-0.2           ],
                    [ 0.1            ],
                    [ 0.2           ],
                    [ 0.1            ]])
    robot.display(q)                
    #TEST THE FOOT JACOBIAN ********************************************     
    def dLF_over_dq(i,q,frameID):
        '''compute jacobian via finite differences'''
        eps = 1e-8
        q1 = q.copy()
        v = np.zeros([7,1])
        v[i,0] = 1
        Mlf1, Mrf1 = robot.get_Mlf_Mrf(q1)
        q2   = se3.integrate(robot.model,q1,v*eps)
        Mlf2, Mrf2 = robot.get_Mlf_Mrf(q2)
        return (Mlf2.translation - Mlf1.translation) / eps
    Jl_fd = np.hstack([dLF_over_dq(0,q,LF),
                       dLF_over_dq(1,q,LF),
                       dLF_over_dq(2,q,LF),
                       dLF_over_dq(3,q,LF),
                       dLF_over_dq(4,q,LF),
                       dLF_over_dq(5,q,LF),
                       dLF_over_dq(6,q,LF)])          
    Jl,Jr = robot.get_Jl_Jr_world(q)
    assert isapprox(Jl_fd, Jl[:3])

    #TEST THE FOOT VELOCITY ********************************************       
    q = np.matrix([[ 0.1           ],
                   [ 0.569638      ],
                   [ np.cos(teta)  ],
                   [ np.sin(teta)  ],
                   [-0.2           ],
                   [ 0.1           ],
                   [ 0.2           ],
                   [ 0.1           ]])
                    
    v = np.matrix([[ 0.1  ],
                   [ 0.2  ],
                   [ 0.3  ],
                   [ 0.2  ],
                   [ 0.1  ],
                   [ 0.2  ],
                   [ 0.3  ]])
    
    q1  = q.copy()
    Mlf1, Mrf1 = robot.get_Mlf_Mrf(q1)
    eps = 1e-8
    q2  = se3.integrate(robot.model,q1,v*eps)
    Mlf2, Mrf2 = robot.get_Mlf_Mrf(q2)
    vlf_fd = (Mlf2.translation - Mlf1.translation)*1/eps
    vlf,vrf = robot.get_vlf_vrf_world(q1,v)
    assert isapprox(vlf.linear, vlf_fd)
    
    #TEST THE FOOT ACCELERATION ****************************************      
    q = np.matrix([[ 0.1           ],
                   [ 0.569638      ],
                   [ np.cos(teta)  ],
                   [ np.sin(teta)  ],
                   [-0.2           ],
                   [ 0.1           ],
                   [ 0.2           ],
                   [ 0.1           ]])
                    
    v = np.matrix([[ 0.11  ],
                   [ 0.12  ],
                   [ 0.13  ],
                   [ 0.14  ],
                   [ 0.15  ],
                   [ 0.26  ],
                   [ 0.17  ]])
                   
    a = np.matrix([[ 0.15  ],
                   [ 0.16  ],
                   [ 0.17  ],
                   [ 0.18  ],
                   [ 0.19  ],
                   [ 0.10  ],
                   [ 0.11  ]])
    q1  = q.copy()
    v1  = v.copy()
    vlf1,vrf1 = robot.get_vlf_vrf_world(q1,v1)
    eps = 1e-8
    q2  = se3.integrate(robot.model,q1,v1*eps)
    v2  = v1 + eps*a
    vlf2,vrf2 = robot.get_vlf_vrf_world(q2,v2)
    alf_fd = (vlf2.linear - vlf1.linear)*1/eps
    Jl,Jr = robot.get_Jl_Jr_world(q1)
    driftLF,driftRF = robot.get_driftLF_driftRF_world(q1,v1)
    
    alf,arf = robot.get_classic_alf_arf(q1,v1,a)
    print "acc pino including CORIOLIS"
    print alf.vector[1:3]
    print "acc finite diff"
    print alf_fd[1:3]
    print "acc Jdv+dJdq"
    print (Jl*a)[1:3] + driftLF.vector[1:3]
    
    assert isapprox(alf.vector[1:3] , alf_fd[1:3])
    assert isapprox((Jl*a)[1:3] + driftLF.vector[1:3] , alf_fd[1:3])
    assert isapprox((Jl*a)[1:3] + driftLF.vector[1:3] , alf.vector[1:3])

    Jam = robot.get_angularMomentumJacobian(q,v)
    try:
        embed()
    except:
        pass
    
    for i in range(0,8):
        for p in np.linspace(0,1,50).tolist()+np.linspace(1,0,50).tolist():
            q[i]=robot.q0[i]+p
            robot.display(q)
            time.sleep(0.01)

