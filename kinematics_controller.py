import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import math
import numpy as np

class RobotArmKinematicsEngine(Node):
    def __init__(self):
        super().__init__('robot_arm_kinematics_engine')
        
        self.declare_parameter('link1_length', 1.0)
        self.declare_parameter('link2_length', 1.0)
        self.declare_parameter('loop_rate', 0.1)
        
        self.l1 = self.get_parameter('link1_length').get_parameter_value().double_value
        self.l2 = self.get_parameter('link2_length').get_parameter_value().double_value
        
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )
        
        self.timer_period = self.get_parameter('loop_rate').get_parameter_value().double_value
        self.timer = self.create_timer(self.timer_period, self.execute_trajectory_step)
        
        self.current_joints = [0.0, 0.0]
        self.time_tracker = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

    def state_callback(self, msg):
        if len(msg.position) >= 2:
            self.current_joints = [msg.position[0], msg.position[1]]

    def calculate_inverse_kinematics(self, x, y):
        dist_sq = x**2 + y**2
        cos_theta2 = (dist_sq - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        
        theta2 = math.acos(cos_theta2)
        
        k1 = self.l1 + self.l2 * cos_theta2
        k2 = self.l2 * math.sin(theta2)
        
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)
        
        return theta1, theta2

    def execute_trajectory_step(self):
        self.time_tracker += self.timer_period
        
        self.target_x = 0.5 + 0.3 * math.cos(self.time_tracker)
        self.target_y = 0.5 + 0.3 * math.sin(self.time_tracker)
        
        try:
            q1, q2 = self.calculate_inverse_kinematics(self.target_x, self.target_y)
            
            msg = JointTrajectory()
            msg.joint_names = ['joint1', 'joint2']
            
            point = JointTrajectoryPoint()
            point.positions = [q1, q2]
            point.time_from_start = Duration(sec=0, nanosec=int(self.timer_period * 1e9))
            
            msg.points.append(point)
            self.trajectory_publisher.publish(msg)
            
            self.get_logger().info(f'Target XY: ({self.target_x:.2f}, {self.target_y:.2f}) -> Joints: ({q1:.2f}, {q2:.2f})')
            
        except ValueError:
            self.get_logger().error('Target position out of reachable workspace')

def main(args=None):
    rclpy.init(args=args)
    engine = RobotArmKinematicsEngine()
    
    try:
        rclpy.spin(engine)
    except KeyboardInterrupt:
        pass
    finally:
        engine.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
