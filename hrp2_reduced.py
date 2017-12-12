import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt

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
        if refresh: display.refresh()

class Hrp2Reduced:
    '''
    Class following robot-wrapper template, with a model, a data and a viewer object.
    Init from the urdf model of HRP-2.
    Display with robot.display. Initial configuration set to self.q0.
    '''
    def __init__(self,urdf,pkgs,loadModel=True):
        '''Typical call with Hrp2Reduced('..../hrp2.udf',['dir/with/mesh/files']).'''
        self.loadHRP(urdf,pkgs,loadModel)
        self.buildModel()

    def loadHRP(self,urdf,pkgs,loadModel):
        '''Internal: load HRP-2 model from URDF.'''
        robot = RobotWrapper( urdf, pkgs, root_joint=se3.JointModelFreeFlyer() )
        robot.initDisplay( loadModel = loadModel)
        if 'viewer' not in robot.__dict__: robot.initDisplay()
        for n in robot.visual_model.geometryObjects: robot.viewer.gui.setVisibility(robot.viewerNodeNames(n),'OFF')

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
        viewer.addBox('world/red'+bodyName, 0.05,0.05,0.05,colorred)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add right ankle spot
        name = "rankle"
        placement = robot.data.oMi[RKNEE_ID].inverse()*robot.data.oMi[RANKLE_ID]
        t = placement.translation;        t[0] = 0;   t[1]=0;     placement.translation=t
        placement.rotation = rotate('y',0)

        modelred.addFrame(se3.Frame(name,jointId,0,placement,se3.FrameType.OP_FRAME))
        viewer.addSphere('world/red'+name, 0.05,colorred)
        visuals.append( Visual('world/red'+name,jointId,placement*se3.SE3(eye(3),np.matrix([0,0,.05]).T)) )
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
        viewer.addBox('world/red'+bodyName, 0.05,0.05,0.05,colorblue)
        visuals.append( Visual('world/red'+bodyName,jointId,se3.SE3.Identity()) )

        # Add left ankle spot
        name = "lankle"
        placement = robot.data.oMi[LKNEE_ID].inverse()*robot.data.oMi[LANKLE_ID]
        t = placement.translation;        t[0] = 0;   t[1]=0;     placement.translation=t
        placement.rotation = rotate('y',0)

        modelred.addFrame(se3.Frame(name,jointId,0,placement,se3.FrameType.OP_FRAME))
        viewer.addSphere('world/red'+name, 0.05,colorblue)
        visuals.append( Visual('world/red'+name,jointId,placement*se3.SE3(eye(3),np.matrix([0,0,.05]).T)) )
        viewer.addBox('world/red'+name+'sole', 0.15,.06,.01,colorblue)
        visuals.append( Visual('world/red'+name+'sole',jointId,placement) )

        self.data = modelred.createData()
        self.q0   =  np.matrix([0,0.569638,1.,0,0,0,0,0]).T

    def display(self,q):
        se3.forwardKinematics(self.model,self.data,q)
        for visual in self.visuals:
            visual.place( self.viewer,self.data.oMi[visual.jointParent] )
            self.viewer.refresh()


if __name__ == '__main__':
    pkg = '/home/nmansard/src/sot_versions/groovy/ros/stacks/hrp2/'
    urdf = '/home/nmansard/src/pinocchio/pinocchio/models/hrp2014.urdf'
    robot = Hrp2Reduced(urdf,[pkg])
    robot.display(robot.q0)
    
