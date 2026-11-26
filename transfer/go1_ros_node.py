"""ROS 2 node: the Go1 policy walks in plain MuJoCo, driven by /cmd_vel.

Same contract as the Project 1 rover: Twist in on /cmd_vel, Odometry out on /odom.
Run from the repo root, inside the `ros` pixi env:
  python transfer/go1_ros_node.py --ros-args -p onnx:=results/flat_s1.onnx -p scene:=scene/scene.xml
Honest limit: /odom is simulator truth (pose from qpos, body-frame velocity from the velocimeter).
A real robot would have to estimate both.
"""
# [permanent robotics] The policy is just another controller behind a standard ROS 2 interface.
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from go1_port import Go1Port

LIMITS = np.array([1.0, 1.0, 0.5])  # vx, vy, wz: the range the policy was trained on
DEADMAN_S = 0.5  # same rule as the rover: no command for 0.5 s -> stand still


class Go1PolicyNode(Node):
    def __init__(self):
        super().__init__("go1_policy")
        onnx = self.declare_parameter("onnx", "policy.onnx").value
        scene = self.declare_parameter("scene", "scene/scene.xml").value
        self.port = Go1Port(scene, onnx)
        self.port.reset()
        self.last_cmd_time = self.get_clock().now()
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self)  # odom -> base_link, so RViz can show the robot moving
        period = self.port.m.opt.timestep * self.port.decimation  # 0.02 s -> 50 Hz
        self.create_timer(period, self.tick)

    def on_cmd(self, msg):
        cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.port.command = np.clip(cmd, -LIMITS, LIMITS)  # never ask outside training range
        self.last_cmd_time = self.get_clock().now()

    def tick(self):
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if age > DEADMAN_S:
            self.port.command = np.zeros(3)
        self.port.step()
        self.publish()

    def publish(self):
        d, now = self.port.d, self.get_clock().now().to_msg()
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, "odom", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, d.qpos[0:3])
        w, x, y, z = d.qpos[3:7]  # MuJoCo quaternion order is w, x, y, z
        q = o.pose.pose.orientation
        q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
        lin = d.sensordata[self.port.s_linvel]
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = map(float, lin)
        self.odom_pub.publish(o)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now, "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, d.qpos[0:3])
        tf.transform.rotation = o.pose.pose.orientation
        self.tf.sendTransform(tf)
        js = JointState()
        js.header.stamp = now
        js.name = self.port.meta["joint_names"]
        js.position = [float(v) for v in d.qpos[self.port.qadr]]
        js.velocity = [float(v) for v in d.qvel[self.port.vadr]]
        self.js_pub.publish(js)


def main():
    rclpy.init()
    rclpy.spin(Go1PolicyNode())


if __name__ == "__main__":
    main()
