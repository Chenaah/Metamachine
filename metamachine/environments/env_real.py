"""
Real Robot Environment using Capybarish for ESP32 Communication.

This module provides the RealMetaMachine class for controlling real robots
via the capybarish NetworkServer API. It receives sensor data from ESP32
modules and sends motor position commands back.

Key Features:
- Ordered module mapping: action[i] -> module_ids[i] from config
- Auto-discovery with validation against expected modules
- Optional Rich dashboard for real-time monitoring
- Full compatibility with simulation configs

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

import copy
import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np
from omegaconf import OmegaConf

from .base import Base

# Import capybarish for ESP32 communication
try:
    from capybarish.pubsub import NetworkServer, Rate
    from capybarish.generated import ReceivedData, SentData
    CAPYBARISH_AVAILABLE = True
except ImportError:
    CAPYBARISH_AVAILABLE = False
    NetworkServer = None
    ReceivedData = None
    SentData = None

# Import dashboard (optional)
try:
    from capybarish.dashboard import MotorDashboard, DashboardConfig
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    MotorDashboard = None
    DashboardConfig = None


def sanitize_dict(d: dict) -> dict:
    """Sanitize dictionary for JSON serialization."""
    result = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, (np.float32, np.float64)):
            result[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            result[key] = int(value)
        else:
            result[key] = value
    return result


class RealMetaMachine(Base):
    """Real robot environment using Capybarish for ESP32 communication.

    This class provides the interface for controlling real robots via
    the NetworkServer pattern. ESP32 modules send sensor data to the server,
    and the server sends motor position commands back.
    
    Module Ordering:
        The `module_ids` config parameter defines BOTH which modules are expected
        AND their order. For example:
            module_ids: [2, 0, 1]
        means:
            - action[0] controls module 2
            - action[1] controls module 0  
            - action[2] controls module 1
        
        This allows flexible mapping between action indices and physical modules.
    
    Network Communication:
        - Receives: SentData (sensor feedback from ESP32 modules)
        - Sends: ReceivedData (motor commands to ESP32 modules)
    
    Config Structure (in YAML):
        ```yaml
        environment:
          mode: real  # or "sim"
          num_envs: 1
        
        real:
          module_ids: [0, 1, 2]     # Expected modules in action order
          listen_port: 6666         # Port to receive sensor data
          command_port: 6667        # Port to send commands
          device_timeout: 2.0       # Seconds before module inactive
          enable_filter: true       # Enable ESP32 low-pass filter
          enable_dashboard: true    # Show Rich dashboard
        
        control:
          num_actions: 3
          kp: 10.0
          kd: 0.5
        ```
    """

    # Default network ports
    DEFAULT_LISTEN_PORT = 6666
    DEFAULT_COMMAND_PORT = 6667

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize the real robot environment.

        Args:
            cfg: Configuration object for the environment
            
        Raises:
            ImportError: If capybarish is not installed
            ValueError: If module_ids not specified in config
        """
        if not CAPYBARISH_AVAILABLE:
            raise ImportError(
                "capybarish is required for real robot control. "
                "Install it with: pip install capybarish"
            )
        
        # Get real robot configuration
        real_cfg = cfg.get("real", {})
        
        # =====================================================================
        # CRITICAL: Get expected module IDs from config
        # =====================================================================
        self.expected_module_ids: List[int] = list(real_cfg.get("module_ids", []))
        if not self.expected_module_ids:
            # Fallback: generate from num_actions if not specified
            num_actions = cfg.control.num_actions
            self.expected_module_ids = list(range(num_actions))
            print(f"[Warning] No module_ids in config. Using default: {self.expected_module_ids}")
        
        # Validate module count matches num_actions
        num_actions = cfg.control.num_actions
        if len(self.expected_module_ids) != num_actions:
            raise ValueError(
                f"module_ids length ({len(self.expected_module_ids)}) must match "
                f"num_actions ({num_actions})"
            )
        
        # Network configuration
        self.listen_port = real_cfg.get("listen_port", self.DEFAULT_LISTEN_PORT)
        self.command_port = real_cfg.get("command_port", self.DEFAULT_COMMAND_PORT)
        self.device_timeout = real_cfg.get("device_timeout", 2.0)
        
        # Control parameters
        self.kp_default = cfg.control.get("kp", 10.0)
        self.kd_default = cfg.control.get("kd", 0.5)
        self.enable_filter = real_cfg.get("enable_filter", True)
        
        # Dashboard configuration
        self.enable_dashboard = real_cfg.get("enable_dashboard", False)
        self.dashboard: Optional[MotorDashboard] = None
        
        # Initialize the network server
        self._init_network_server()
        
        # =====================================================================
        # Module tracking with ordered mapping
        # =====================================================================
        # Maps: module_id -> IP address (discovered dynamically)
        self.module_to_ip: Dict[int, str] = {}
        # Maps: IP address -> module_id
        self.ip_to_module: Dict[str, int] = {}
        # Latest data from each module
        self.module_data: Dict[int, SentData] = {}
        # Set of connected modules
        self.connected_modules: Set[int] = set()
        
        # Control state
        self.motor_enabled = False
        self.last_motor_com_time = time.time()
        self.compute_time = 0.0
        self.send_dt = 0.0
        
        # Per-motor gains
        self.kps = np.ones(num_actions) * self.kp_default
        self.kds = np.ones(num_actions) * self.kd_default
        
        # Statistics
        self.cmd_count = 0
        self.fb_count = 0
        self.start_time = time.time()
        
        # Initialize parent class
        super(RealMetaMachine, self).__init__(cfg)
        
        # Setup logging
        self.log_dir = self.cfg.logging.get("robot_data_dir", None)
        self.log_file = None
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            log_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
            self.log_file = open(os.path.join(self.log_dir, log_filename), "w")
        
        # Observable data cache
        self.observable_data: Dict[str, Any] = {}
        
        # Initialize dashboard if enabled
        if self.enable_dashboard:
            self._init_dashboard()
        
        self._print_startup_info()

    def _print_startup_info(self) -> None:
        """Print startup information."""
        print("=" * 60)
        print("RealMetaMachine Initialized")
        print("=" * 60)
        print(f"  Expected modules: {self.expected_module_ids}")
        print(f"  Num actions: {len(self.expected_module_ids)}")
        print(f"  Listen port: {self.listen_port}")
        print(f"  Command port: {self.command_port}")
        print(f"  Kp: {self.kp_default}, Kd: {self.kd_default}")
        print(f"  Dashboard: {'enabled' if self.enable_dashboard else 'disabled'}")
        print("=" * 60)
        print("\nAction -> Module mapping:")
        for i, mod_id in enumerate(self.expected_module_ids):
            print(f"  action[{i}] -> module {mod_id}")
        print("\nWaiting for ESP32 modules to connect...")

    def _init_network_server(self) -> None:
        """Initialize the NetworkServer for ESP32 communication."""
        self.server = NetworkServer(
            recv_type=SentData,
            send_type=ReceivedData,
            recv_port=self.listen_port,
            send_port=self.command_port,
            callback=self._on_module_feedback,
            timeout_sec=self.device_timeout,
        )

    def _init_dashboard(self) -> None:
        """Initialize the Rich dashboard for real-time monitoring."""
        if not DASHBOARD_AVAILABLE:
            print("[Warning] Dashboard not available. Install rich: pip install rich")
            self.enable_dashboard = False
            return
        
        try:
            config = DashboardConfig(
                title="RealMetaMachine Controller",
                refresh_rate=20,
                timeout_sec=self.device_timeout,
            )
            self.dashboard = MotorDashboard(config)
            self.dashboard.start()
        except Exception as e:
            print(f"[Warning] Failed to start dashboard: {e}")
            self.dashboard = None
            self.enable_dashboard = False

    def _on_module_feedback(self, msg: SentData, sender_ip: str) -> None:
        """Callback when sensor data is received from an ESP32 module.
        
        Args:
            msg: The SentData message containing sensor readings
            sender_ip: IP address of the sender
        """
        module_id = msg.module_id
        self.fb_count += 1
        
        # Check if this is an expected module
        if module_id not in self.expected_module_ids:
            # Unexpected module - log warning once
            if module_id not in self.connected_modules:
                print(f"[Warning] Unexpected module {module_id} at {sender_ip} "
                      f"(expected: {self.expected_module_ids})")
            return
        
        # Track new module connections
        if module_id not in self.connected_modules:
            self.connected_modules.add(module_id)
            self.module_to_ip[module_id] = sender_ip
            self.ip_to_module[sender_ip] = module_id
            
            # Find action index for this module
            action_idx = self.expected_module_ids.index(module_id)
            print(f"[Connected] Module {module_id} at {sender_ip} -> action[{action_idx}]")
            
            # Check if all modules are now connected
            if self.all_modules_connected():
                print(f"[Ready] All {len(self.expected_module_ids)} modules connected!")
        
        # Update IP if changed (module moved to different network)
        elif self.module_to_ip.get(module_id) != sender_ip:
            old_ip = self.module_to_ip.get(module_id)
            self.module_to_ip[module_id] = sender_ip
            if old_ip:
                self.ip_to_module.pop(old_ip, None)
            self.ip_to_module[sender_ip] = module_id
            print(f"[Update] Module {module_id} IP changed: {old_ip} -> {sender_ip}")
        
        # Store latest data
        self.module_data[module_id] = msg
        
        # Update dashboard if enabled
        if self.dashboard is not None:
            self._update_dashboard_motor(module_id, msg, sender_ip)

    def _update_dashboard_motor(self, module_id: int, msg: SentData, sender_ip: str) -> None:
        """Update dashboard with motor data."""
        if self.dashboard is None:
            return
        
        motor = msg.motor if hasattr(msg, 'motor') else None
        
        # Build error string
        error_str = ""
        if hasattr(msg, 'error') and msg.error:
            err = msg.error
            if hasattr(err, 'reset_reason0') and hasattr(err, 'reset_reason1'):
                if err.reset_reason0 != 0 or err.reset_reason1 != 0:
                    error_str = f"Reset: {err.reset_reason0}/{err.reset_reason1}"
        
        # Get action index for this module
        try:
            action_idx = self.expected_module_ids.index(module_id)
        except ValueError:
            action_idx = -1
        
        self.dashboard.update_motor(
            address=sender_ip,
            name=f"M{module_id}->A{action_idx}",
            position=motor.pos if motor else 0.0,
            velocity=motor.vel if motor else 0.0,
            torque=motor.torque if motor else 0.0,
            voltage=motor.voltage if motor else 0.0,
            current=motor.current if motor else 0.0,
            mode="Running" if self.motor_enabled else "Idle",
            switch=self.motor_enabled,
            error=error_str,
        )

    def all_modules_connected(self) -> bool:
        """Check if all expected modules are connected.
        
        Returns:
            bool: True if all expected modules are connected
        """
        return all(mid in self.connected_modules for mid in self.expected_module_ids)

    def ready(self) -> bool:
        """Check if the robot system is ready for control.
        
        The system is ready when all expected modules are connected
        and actively sending data.
        
        Returns:
            bool: True if all expected modules are active
        """
        # Check all expected modules are connected
        if not self.all_modules_connected():
            return False
        
        # Check all modules are in active devices (recently seen)
        active_ips = set(self.server.active_devices.keys())
        for module_id in self.expected_module_ids:
            ip = self.module_to_ip.get(module_id)
            if ip not in active_ips:
                return False
        
        return True

    def get_missing_modules(self) -> List[int]:
        """Get list of expected modules that are not yet connected.
        
        Returns:
            List of module IDs that are expected but not connected
        """
        return [mid for mid in self.expected_module_ids if mid not in self.connected_modules]

    def get_inactive_modules(self) -> List[int]:
        """Get list of modules that are connected but not actively sending.
        
        Returns:
            List of module IDs that are connected but inactive
        """
        active_ips = set(self.server.active_devices.keys())
        inactive = []
        for module_id in self.expected_module_ids:
            if module_id in self.connected_modules:
                ip = self.module_to_ip.get(module_id)
                if ip not in active_ips:
                    inactive.append(module_id)
        return inactive

    def update_config(self, cfg: OmegaConf) -> None:
        """Update environment configuration."""
        self.cfg = cfg
        
        # Update control parameters
        self.kp_default = cfg.control.get("kp", self.kp_default)
        self.kd_default = cfg.control.get("kd", self.kd_default)
        
        # Update gains arrays
        num_actions = len(self.expected_module_ids)
        self.kps = np.ones(num_actions) * self.kp_default
        self.kds = np.ones(num_actions) * self.kd_default
        
        self._initialize_components()

    def _log_data(self) -> None:
        """Log observable data from the real robot."""
        if self.log_file is not None and self.observable_data:
            self.log_file.write(
                json.dumps(sanitize_dict(copy.deepcopy(self.observable_data))) + "\n"
            )
            self.log_file.flush()

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return not self.ready()

    def receive_module_data(self) -> int:
        """Process incoming sensor data from all ESP32 modules."""
        return self.server.spin_once()

    def _get_observable_data(self) -> Dict[str, Any]:
        """Get current observable state data from the real robot.

        Returns data ordered according to expected_module_ids:
            - dof_pos[i] corresponds to expected_module_ids[i]
            - dof_vel[i] corresponds to expected_module_ids[i]
            etc.
        """
        # Process incoming messages
        self.receive_module_data()
        
        num_actions = len(self.expected_module_ids)
        
        # Initialize arrays in action order
        dof_pos = np.zeros(num_actions)
        dof_vel = np.zeros(num_actions)
        
        # Per-module data lists (in action order)
        imu_quats = []
        imu_gyros = []
        imu_accels = []
        projected_gravities = []
        
        # Collect data for each expected module IN ORDER
        for action_idx, module_id in enumerate(self.expected_module_ids):
            if module_id in self.module_data:
                data = self.module_data[module_id]
                
                # Motor data
                dof_pos[action_idx] = data.motor.pos
                dof_vel[action_idx] = data.motor.vel
                
                # IMU quaternion [x, y, z, w]
                quat = np.array([
                    data.imu.quaternion.x,
                    data.imu.quaternion.y,
                    data.imu.quaternion.z,
                    data.imu.quaternion.w
                ])
                imu_quats.append(quat)
                
                # IMU angular velocity (gyro)
                gyro = np.array([
                    data.imu.omega.x,
                    data.imu.omega.y,
                    data.imu.omega.z
                ])
                imu_gyros.append(gyro)
                
                # IMU acceleration
                accel = np.array([
                    data.imu.acceleration.x,
                    data.imu.acceleration.y,
                    data.imu.acceleration.z
                ])
                imu_accels.append(accel)
                
                # Compute projected gravity from quaternion
                projected_gravity = self._rotate_vector_by_quat(
                    np.array([0, 0, -1]), quat
                )
                projected_gravities.append(projected_gravity)
            else:
                # Module not yet received data - use defaults
                imu_quats.append(np.array([0, 0, 0, 1]))
                imu_gyros.append(np.zeros(3))
                imu_accels.append(np.array([0, 0, -9.81]))
                projected_gravities.append(np.array([0, 0, -1]))
        
        # Use first module's data for global values (main module)
        main_gravity = projected_gravities[0] if projected_gravities else np.array([0, 0, -1])
        main_gyro = imu_gyros[0] if imu_gyros else np.zeros(3)
        
        self.observable_data = {
            # Joint state (ordered by action index)
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            
            # Global (torso) state - use main module (first in order)
            "projected_gravity": main_gravity,
            "ang_vel_body": main_gyro,
            "vel_body": np.zeros(3),  # Not available from IMU
            
            # Per-module data (ordered by action index)
            "projected_gravities": projected_gravities,
            "gyros": imu_gyros,
            "quats": imu_quats,
            "accs": imu_accels,
            
            # Metadata
            "timestamp": time.time(),
            "module_order": self.expected_module_ids,
        }
        
        # Update dashboard performance
        if self.dashboard is not None:
            elapsed = time.time() - self.start_time
            self.dashboard.set_status("Cmd/Fb", f"{self.cmd_count}/{self.fb_count}")
            self.dashboard.set_status("Modules", 
                f"{len(self.connected_modules)}/{len(self.expected_module_ids)}")
            self.dashboard.update()
        
        # Log data if enabled
        if self.cfg.logging.get("log_raw_data", False):
            self._log_data()

        # print(f"Observable data: {self.observable_data}")
        
        return self.observable_data

    def _rotate_vector_by_quat(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        qx, qy, qz, qw = q
        q_vec = np.array([qx, qy, qz])
        cross1 = np.cross(q_vec, v)
        cross2 = np.cross(q_vec, cross1)
        return v + 2.0 * qw * cross1 + 2.0 * cross2

    def _wait_until_motor_on(self) -> None:
        """Wait until all expected robot modules are ready."""
        wait_count = 0
        rate = Rate(50.0)  # 50 Hz check rate
        
        while not self.ready():
            # Process incoming messages
            self.receive_module_data()
            
            # Send keepalive to connected modules
            if self.connected_modules:
                zeros = np.zeros(len(self.expected_module_ids))
                self._send_motor_commands(
                    positions=zeros,
                    velocities=zeros,
                    kps=np.zeros(len(self.expected_module_ids)),
                    kds=np.zeros(len(self.expected_module_ids)),
                    enable=False
                )
            
            # Print status periodically
            if wait_count == 0 or wait_count % 100 == 0:
                missing = self.get_missing_modules()
                inactive = self.get_inactive_modules()
                connected = len(self.connected_modules)
                total = len(self.expected_module_ids)
                
                status = f"Waiting for modules... ({connected}/{total} connected)"
                if missing:
                    status += f" Missing: {missing}"
                if inactive:
                    status += f" Inactive: {inactive}"
                print(status)
            
            self._check_input()
            rate.sleep()
            wait_count += 1

    def _perform_action(
        self, 
        pos: np.ndarray, 
        vel: Optional[np.ndarray] = None, 
        kps: Optional[np.ndarray] = None, 
        kds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Execute action on the real robot.

        Args:
            pos: Position commands (radians), indexed by action order
            vel: Velocity commands (rad/s, optional)
            kps: Position gains (optional)
            kds: Derivative gains (optional)

        Note:
            pos[i] is sent to expected_module_ids[i]
        """
        if kps is not None:
            self.kps = kps
        if kds is not None:
            self.kds = kds
        
        if vel is None:
            vel = np.zeros_like(pos)
        
        # Send commands to all modules
        sent = self._send_motor_commands(
            positions=pos,
            velocities=vel,
            kps=self.kps,
            kds=self.kds,
            enable=self.motor_enabled
        )
        self.cmd_count += sent
        
        # Track timing
        self.compute_time = time.time() - self.t0
        
        # Wait for control timestep
        dt = self.cfg.control.dt
        while time.time() - self.t0 < dt:
            pass
        
        self.send_dt = time.time() - self.t0
        self.t0 = time.time()
        
        return {
            "compute_time": self.compute_time,
            "send_dt": self.send_dt,
            "commands_sent": sent,
        }

    def _send_motor_commands(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        kps: np.ndarray,
        kds: np.ndarray,
        enable: bool = True
    ) -> int:
        """Send motor commands to all expected ESP32 modules.
        
        Commands are sent in the order defined by expected_module_ids:
            positions[i] -> expected_module_ids[i]
        """
        sent_count = 0
        current_time = time.time()
        
        # Send commands to each expected module in order
        for action_idx, module_id in enumerate(self.expected_module_ids):
            if module_id not in self.module_to_ip:
                continue  # Module not yet connected
            
            ip = self.module_to_ip[module_id]
            
            # Create command message
            cmd = ReceivedData(
                target=float(positions[action_idx]),
                target_vel=float(velocities[action_idx]) if action_idx < len(velocities) else 0.0,
                kp=float(kps[action_idx]) if action_idx < len(kps) else self.kp_default,
                kd=float(kds[action_idx]) if action_idx < len(kds) else self.kd_default,
                enable_filter=1 if self.enable_filter else 0,
                switch_=1 if enable else 0,
                calibrate=0,
                restart=0,
                timestamp=current_time,
            )
            
            if self.server.send_to(ip, cmd):
                sent_count += 1
        
        self.last_motor_com_time = current_time
        return sent_count

    def _reset_robot(self) -> None:
        """Reset real robot to initial state."""
        self._wait_until_motor_on()
        
        num_actions = len(self.expected_module_ids)
        zero_pos = np.zeros(num_actions)
        
        default_dof_pos = self.cfg.control.get("default_dof_pos", None)
        if default_dof_pos is not None:
            init_pos = np.array(default_dof_pos)
        else:
            init_pos = zero_pos
        
        self._send_motor_commands(
            positions=init_pos,
            velocities=zero_pos,
            kps=self.kps,
            kds=self.kds,
            enable=self.motor_enabled
        )
        
        time.sleep(0.5)

    def _handle_input(self) -> None:
        """Override base class to handle real robot keyboard input.
        
        This is called by Base.step() on every step to process keyboard input.
        For real robots, we handle enable/disable/restart/calibrate commands.
        """
        self._check_input()

    def _check_input(self) -> None:
        """Handle keyboard input for real robot control."""
        if hasattr(self, 'kb') and self.kb.kbhit():
            self.input_key = self.kb.getch()
            
            if self.input_key == "e":
                self._enable_motor()
            elif self.input_key == "d":
                self._disable_motor()
            elif self.input_key == "r":
                self._restart_motor()
            elif self.input_key == "c":
                self._calibrate_motor()
            
            if time.time() - self.last_motor_com_time > 0.5:
                self._reset_motor_commands()

    def _enable_motor(self) -> None:
        """Enable all motors."""
        print("[CMD] Enabling motors...")
        self.motor_enabled = True
        self._send_enable_command(enable=True)
        if self.dashboard:
            self.dashboard.set_switch(True)

    def _disable_motor(self) -> None:
        """Disable all motors."""
        print("[CMD] Disabling motors...")
        self.motor_enabled = False
        self._send_enable_command(enable=False)
        if self.dashboard:
            self.dashboard.set_switch(False)

    def _restart_motor(self, module_id: str = "auto") -> None:
        """Restart motor(s)."""
        print(f"[CMD] Restarting motors ({module_id})...")
        self._send_special_command(restart=1)
        self.last_motor_com_time = time.time()

    def _calibrate_motor(self, module_id: str = "auto") -> None:
        """Calibrate motor(s)."""
        print(f"[CMD] Calibrating motors ({module_id})...")
        self._send_special_command(calibrate=1)

    def _reset_motor_commands(self) -> None:
        """Reset motor commands after timeout."""
        zeros = np.zeros(len(self.expected_module_ids))
        self._send_motor_commands(
            positions=zeros,
            velocities=zeros,
            kps=zeros,
            kds=zeros,
            enable=False
        )

    def _send_enable_command(self, enable: bool) -> None:
        """Send enable/disable command to all modules."""
        current_time = time.time()
        
        for module_id in self.expected_module_ids:
            if module_id not in self.module_to_ip:
                continue
            
            ip = self.module_to_ip[module_id]
            cmd = ReceivedData(
                target=0.0,
                target_vel=0.0,
                kp=self.kp_default,
                kd=self.kd_default,
                enable_filter=1 if self.enable_filter else 0,
                switch_=1 if enable else 0,
                calibrate=0,
                restart=0,
                timestamp=current_time,
            )
            self.server.send_to(ip, cmd)

    def _send_special_command(self, calibrate: int = 0, restart: int = 0) -> None:
        """Send special command (calibrate/restart) to all modules."""
        current_time = time.time()
        
        for module_id in self.expected_module_ids:
            if module_id not in self.module_to_ip:
                continue
            
            ip = self.module_to_ip[module_id]
            cmd = ReceivedData(
                target=0.0,
                target_vel=0.0,
                kp=self.kp_default,
                kd=self.kd_default,
                enable_filter=1 if self.enable_filter else 0,
                switch_=0,
                calibrate=calibrate,
                restart=restart,
                timestamp=current_time,
            )
            self.server.send_to(ip, cmd)

    def close(self) -> None:
        """Clean up resources."""
        self._disable_motor()
        
        if self.dashboard is not None:
            try:
                self.dashboard.stop()
            except Exception:
                pass
        
        if hasattr(self, 'server'):
            self.server.close()
        
        if self.log_file is not None:
            self.log_file.close()
        
        super().close()
        
        # Print summary
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("RealMetaMachine Summary")
        print("=" * 60)
        print(f"  Runtime: {elapsed:.1f}s")
        print(f"  Commands sent: {self.cmd_count}")
        print(f"  Feedback received: {self.fb_count}")
        print(f"  Modules: {list(self.connected_modules)}")
        print("=" * 60)

    # =========================================================================
    # Properties and Status Methods
    # =========================================================================

    @property
    def num_modules(self) -> int:
        """Get number of connected modules."""
        return len(self.connected_modules)

    @property
    def active_modules(self) -> Dict[str, Any]:
        """Get information about active modules."""
        return self.server.active_devices

    def get_module_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all modules, ordered by expected_module_ids."""
        status = {}
        for module_id in self.expected_module_ids:
            if module_id in self.module_data:
                data = self.module_data[module_id]
                status[module_id] = {
                    "connected": True,
                    "action_index": self.expected_module_ids.index(module_id),
                    "ip": self.module_to_ip.get(module_id, "unknown"),
                    "pos": data.motor.pos,
                    "vel": data.motor.vel,
                    "torque": data.motor.torque,
                    "temperature": data.motor.temperature,
                    "voltage": data.motor.voltage,
                    "current": data.motor.current,
                }
            else:
                status[module_id] = {
                    "connected": False,
                    "action_index": self.expected_module_ids.index(module_id),
                }
        return status

    def print_status(self) -> None:
        """Print current status of all modules."""
        print("\n" + "=" * 60)
        print("Module Status")
        print("=" * 60)
        
        for i, module_id in enumerate(self.expected_module_ids):
            status = "✓" if module_id in self.connected_modules else "✗"
            ip = self.module_to_ip.get(module_id, "not connected")
            
            if module_id in self.module_data:
                data = self.module_data[module_id]
                pos = data.motor.pos
                vel = data.motor.vel
                print(f"  [{status}] action[{i}] -> module {module_id} @ {ip}")
                print(f"        pos={pos:+.3f}, vel={vel:+.3f}")
            else:
                print(f"  [{status}] action[{i}] -> module {module_id} @ {ip}")
        
        print("=" * 60)
