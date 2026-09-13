#!/usr/bin/env python3
"""Autonomous gate navigation for the probation Unity/ROS2 simulation.

High-level strategy
-------------------
1. Wait for the MAVROS services, then arm the vehicle and switch it to
   ``GUIDED`` mode. Velocity commands are ignored in any other mode, and an
   unarmed vehicle will not move.
2. SEARCH: rotate in place until the front camera detects the gate. If several
   full rotations produce no detection, start descending slowly while rotating
   so we also scan lower depths.
3. TRACK: keep the gate centred in the camera frame using yaw (horizontal) and
   depth (vertical) feedback, and move forward once roughly aligned.
4. PASS: once the gate is large enough in the image (we are close), drive
   straight through it for a fixed duration, then stop.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Float64
from vision_msgs.msg import BoundingBoxArray


class GateNavigator(Node):
    """A ROS2 node that drives the AUV through the gate autonomously."""

    # ------------------------------------------------------------------
    # Control tuning constants. These are reasonable starting points, but
    # you will likely need to adjust them against the real simulator.
    # ------------------------------------------------------------------
    CONTROL_HZ = 10.0                 # control-loop rate (Hz)
    SEARCH_YAW_RATE = 0.4             # rad/s while rotating to find the gate
    SEARCH_SCANS_BEFORE_DESCEND = 2.0 # full rotations before descending
    SEARCH_DESCEND_RATE = -0.1        # m/s while searching low (negative = down)
    YAW_GAIN = 1.2                    # rad/s per unit horizontal error
    VERTICAL_GAIN = 0.5               # m/s per unit vertical error
    FORWARD_SPEED = 0.6               # m/s used for approach and pass
    CENTER_TOL = 0.08                 # |x - 0.5| below this counts as centred
    PASS_BOX_WIDTH = 0.45             # gate box width at which we commit to pass
    # GATE_CONF_THRESHOLD was removed during debugging; re-add it in
    # _on_detections once you know the detector's confidence range.
    GATE_LOST_TIMEOUT = 1.0           # s without a gate box before "lost"
    PASS_DURATION = 5.0               # s to keep driving forward through the gate

    def __init__(self):
        super().__init__('gate_navigator')

        # Latest sensor values.
        self.heading = None        # compass heading in degrees (0..360)
        self.rel_alt = None        # relative altitude / depth in metres
        self.armed = None          # bool from /mavros/state
        self.mode = None           # string from /mavros/state
        self.gate_box = None       # most recent gate BoundingBox (or None)
        self.obstacle_box = None   # most recent obstacle BoundingBox (or None)
        self._last_gate_time = None
        self._seen_labels = set()

        # State machine.
        self.state = 'WAIT_READY'
        self._pass_start_time = None

        # Service-call bookkeeping (so we don't spam requests every tick).
        self._arm_pending = False
        self._mode_pending = False

        # Search bookkeeping (rotation accumulated from the heading topic).
        self._search_heading_start = None
        self._search_heading_last = None
        self._search_rotation = 0.0

        # Status logging throttle.
        self._last_status_log = None
        self._last_detection_log = None

        # Output: velocity commands that actually move the vehicle.
        self.cmd_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # Inputs.
        self.create_subscription(
            BoundingBoxArray,
            '/main_camera/detection/bounding_boxes',
            self._on_detections,
            10)
        self.create_subscription(
            Float64, '/mavros/global_position/compass_hdg',
            self._on_heading, 10)
        self.create_subscription(
            Float64, '/mavros/global_position/rel_alt',
            self._on_altitude, 10)
        self.create_subscription(
            State, '/mavros/state',
            self._on_state, 10)

        # Services for arming and mode switching.
        self._arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self._set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')

        # Main control loop.
        self.create_timer(1.0 / self.CONTROL_HZ, self.control_loop)

        self.get_logger().info('GateNavigator started')

    # ------------------------------------------------------------------
    # Subscription callbacks: just store the latest value.
    # ------------------------------------------------------------------
    def _on_heading(self, msg: Float64):
        self.heading = msg.data

    def _on_altitude(self, msg: Float64):
        self.rel_alt = msg.data

    def _on_state(self, msg: State):
        self.armed = msg.armed
        self.mode = msg.mode

    def _on_detections(self, msg: BoundingBoxArray):
        gate = None
        obstacle = None

        for bb in msg.bounding_boxes:
            name = bb.label_name.lower()

            # Log any class name we haven't seen yet. This is the single most
            # important clue for diagnosing "the robot just spins" — it tells us
            # the detector's actual label strings.
            if bb.label_name not in self._seen_labels:
                self._seen_labels.add(bb.label_name)
                self.get_logger().info(
                    f"New detection label: {bb.label_name!r} (id={bb.label_id})")

            if 'gate' in name:
                if gate is None or bb.conf > gate.conf:
                    gate = bb
            elif 'flare' in name or 'obstacle' in name:
                if obstacle is None or bb.conf > obstacle.conf:
                    obstacle = bb

        # Only *update* the gate when we actually see one. This lets us tolerate
        # a few empty frames without immediately giving up (imperfect detection).
        if gate is not None:
            self.gate_box = gate
            self._last_gate_time = self.get_clock().now()

        self.obstacle_box = obstacle

        # Throttled debug dump so we can see exactly what the camera returns.
        now = self.get_clock().now()
        if (self._last_detection_log is None
                or (now - self._last_detection_log).nanoseconds / 1e9 >= 2.0):
            self._last_detection_log = now
            boxes = ', '.join(
                f"{bb.label_name}(conf={bb.conf:.2f}, x={bb.x:.2f}, "
                f"y={bb.y:.2f}, w={bb.w:.2f})"
                for bb in msg.bounding_boxes)
            self.get_logger().info(
                f'got {len(msg.bounding_boxes)} detection(s): {boxes}')

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def _publish(self, twist: Twist):
        self.cmd_pub.publish(twist)

    @staticmethod
    def _wrap_degrees(angle: float) -> float:
        """Wrap an angle difference into [-180, 180)."""
        return (angle + 180.0) % 360.0 - 180.0

    def _set_state(self, new_state: str):
        if new_state == self.state:
            return
        self.get_logger().info(f'State: {self.state} -> {new_state}')
        self.state = new_state

        if new_state == 'SEARCH':
            self._search_heading_start = None
            self._search_heading_last = None
            self._search_rotation = 0.0
        elif new_state == 'PASS':
            self._pass_start_time = None

    def _gate_is_visible(self) -> bool:
        if self.gate_box is None or self._last_gate_time is None:
            return False
        age = (self.get_clock().now() - self._last_gate_time).nanoseconds / 1e9
        return age <= self.GATE_LOST_TIMEOUT

    def _obstacle_blocking(self) -> bool:
        if self.obstacle_box is None:
            return False
        bb = self.obstacle_box
        # Only react when an obstacle is roughly ahead and reasonably large.
        return bb.w > 0.25 and abs(bb.x - 0.5) < 0.35

    def _update_search_rotation(self):
        if self.heading is None:
            return
        if self._search_heading_last is None:
            self._search_heading_last = self.heading
            return
        delta = self._wrap_degrees(self.heading - self._search_heading_last)
        self._search_rotation += delta
        self._search_heading_last = self.heading

    # ------------------------------------------------------------------
    # Arming and mode switching.
    # ------------------------------------------------------------------
    def _ensure_services(self):
        if (not self._arming_cli.wait_for_service(timeout_sec=0.0)
                or not self._set_mode_cli.wait_for_service(timeout_sec=0.0)):
            return

        if self.armed is False and not self._arm_pending:
            self._arm_pending = True
            req = CommandBool.Request()
            req.value = True
            self.get_logger().info('Requesting ARM')
            self._arming_cli.call_async(req).add_done_callback(self._arm_done)

        if self.mode != 'GUIDED' and not self._mode_pending:
            self._mode_pending = True
            req = SetMode.Request()
            req.base_mode = 0
            req.custom_mode = 'GUIDED'
            self.get_logger().info('Requesting GUIDED mode')
            self._set_mode_cli.call_async(req).add_done_callback(self._mode_done)

    def _arm_done(self, future):
        self._arm_pending = False
        try:
            resp = future.result()
            self.get_logger().info(f'Arm response: success={resp.success}')
        except Exception as exc:  # noqa: BLE001 - retry on any failure
            self.get_logger().warning(f'Arm call failed: {exc}')

    def _mode_done(self, future):
        self._mode_pending = False
        try:
            resp = future.result()
            self.get_logger().info(f'SetMode response: mode_sent={resp.mode_sent}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'SetMode call failed: {exc}')

    def _ready(self) -> bool:
        self._ensure_services()
        return self.armed is True and self.mode == 'GUIDED'

    # ------------------------------------------------------------------
    # Main control loop (runs at CONTROL_HZ).
    # ------------------------------------------------------------------
    def control_loop(self):
        if not self._ready():
            self._publish(Twist())  # hold still until armed + GUIDED
            return

        now = self.get_clock().now()

        if self.state == 'WAIT_READY':
            self._set_state('SEARCH')

        if self.state == 'SEARCH':
            self._control_search()
        elif self.state == 'TRACK':
            self._control_track()
        elif self.state == 'PASS':
            self._control_pass(now)
        else:  # DONE (or any future state)
            self._publish(Twist())

        # Periodic status log for observability in the terminal / Foxglove.
        if (self._last_status_log is None
                or (now - self._last_status_log).nanoseconds / 1e9 >= 5.0):
            self._last_status_log = now
            self.get_logger().info(
                f'state={self.state} heading={self.heading} '
                f'rel_alt={self.rel_alt} gate={self.gate_box is not None} '
                f'obstacle={self.obstacle_box is not None}')

    def _control_search(self):
        if self._gate_is_visible():
            self._set_state('TRACK')
            return

        twist = Twist()
        twist.angular.z = self.SEARCH_YAW_RATE

        self._update_search_rotation()
        full_turns = abs(self._search_rotation) / 360.0
        if full_turns >= self.SEARCH_SCANS_BEFORE_DESCEND:
            twist.linear.z = self.SEARCH_DESCEND_RATE

        self._publish(twist)

    def _control_track(self):
        if not self._gate_is_visible():
            self._set_state('SEARCH')
            return

        gate = self.gate_box

        # Optional obstacle avoidance (bonus, not required by the task).
        if self._obstacle_blocking():
            twist = Twist()
            # Steer away from whichever side the obstacle is on.
            twist.angular.z = (
                self.SEARCH_YAW_RATE
                if self.obstacle_box.x >= 0.5 else -self.SEARCH_YAW_RATE)
            self._publish(twist)
            return

        err_x = gate.x - 0.5   # >0 means gate is right of image centre
        err_y = 0.5 - gate.y   # >0 means gate is above image centre

        twist = Twist()
        twist.angular.z = -self.YAW_GAIN * err_x
        twist.linear.z = self.VERTICAL_GAIN * err_y

        if abs(err_x) < self.CENTER_TOL:
            if gate.w >= self.PASS_BOX_WIDTH:
                self._set_state('PASS')
            else:
                twist.linear.x = self.FORWARD_SPEED * 0.5

        self._publish(twist)

    def _control_pass(self, now):
        if self._pass_start_time is None:
            self._pass_start_time = now

        elapsed = (now - self._pass_start_time).nanoseconds / 1e9

        twist = Twist()
        twist.linear.x = self.FORWARD_SPEED

        # Keep steering toward the gate while driving through.
        if self._gate_is_visible():
            gate = self.gate_box
            twist.angular.z = -self.YAW_GAIN * (gate.x - 0.5)
            twist.linear.z = self.VERTICAL_GAIN * (0.5 - gate.y)

        if elapsed >= self.PASS_DURATION:
            self._set_state('DONE')

        self._publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
