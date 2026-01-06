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
    from capybarish.generated import MotorCommand, SensorData
    CAPYBARISH_AVAILABLE = True
except ImportError:
    CAPYBARISH_AVAILABLE = False
    NetworkServer = None
    MotorCommand = None
    SensorData = None

# Import dashboard (optional)
try:
    from capybarish.dashboard import MotorDashboard, DashboardConfig, RLDashboard, RLDashboardConfig
    DASHBOARD_AVAILABLE = True
    RL_DASHBOARD_AVAILABLE = True
except ImportError:
    try:
        from capybarish.dashboard import MotorDashboard, DashboardConfig
        DASHBOARD_AVAILABLE = True
        RL_DASHBOARD_AVAILABLE = False
        RLDashboard = None
        RLDashboardConfig = None
    except ImportError:
        DASHBOARD_AVAILABLE = False
        RL_DASHBOARD_AVAILABLE = False
        MotorDashboard = None
        DashboardConfig = None
        RLDashboard = None
        RLDashboardConfig = None


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
    
    Module Types:
        1. Active modules (module_ids): Receive motor commands
           - action[i] is sent to module_ids[i]
        2. Sensor modules (sensor_module_ids): No motor commands, sensor data only
           - Useful for dedicated sensors (e.g., distance ranging module)
    
    Module Ordering:
        The `module_ids` config parameter defines BOTH which active modules are 
        expected AND their order. For example:
            module_ids: [2, 0, 1]
        means:
            - action[0] controls module 2
            - action[1] controls module 0  
            - action[2] controls module 1
        
        This allows flexible mapping between action indices and physical modules.
    
    Global State Sources:
        Configure which modules provide global/main sensor data via `sources`:
        - main_imu: Module ID for main quat/gyro (default: first module_id)
        - goal_distance: Module ID for global goal_distance (default: null)
    
    Network Communication:
        - Receives: SensorData (sensor feedback from ESP32 modules)
        - Sends: MotorCommand (motor commands to ESP32 modules)
    
    Config Structure (in YAML):
        ```yaml
        environment:
          mode: real  # or "sim"
          num_envs: 1
        
        real:
          # Active modules (receive motor commands)
          module_ids: [0, 1, 2]     # action[i] -> module_ids[i]
          
          # Sensor-only modules (optional)
          sensor_module_ids: [100]  # No actions, sensor data only
          
          # Global state sources (optional)
          sources:
            main_imu: 0           # Module for main quat/gyro
            goal_distance: 100    # Module for global goal_distance
          
          # Network configuration
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
        # Module Configuration
        # =====================================================================
        
        # Active modules (receive motor commands)
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
        
        # Sensor-only modules (no motor commands, just provide sensor data)
        self.sensor_module_ids: List[int] = list(real_cfg.get("sensor_module_ids", []))
        
        # All expected modules (active + sensor)
        self.all_expected_module_ids: Set[int] = set(self.expected_module_ids) | set(self.sensor_module_ids)
        
        # =====================================================================
        # Global State Sources Configuration
        # =====================================================================
        sources_cfg = real_cfg.get("sources", {})
        
        # Main IMU module (default: first active module)
        self.main_imu_module_id: int = sources_cfg.get("main_imu", self.expected_module_ids[0])
        if self.main_imu_module_id not in self.all_expected_module_ids:
            raise ValueError(
                f"main_imu module {self.main_imu_module_id} not in module_ids or sensor_module_ids"
            )
        
        # Goal distance source module (default: None = no global goal_distance)
        self.goal_distance_module_id: Optional[int] = sources_cfg.get("goal_distance", None)
        if self.goal_distance_module_id is not None and self.goal_distance_module_id not in self.all_expected_module_ids:
            raise ValueError(
                f"goal_distance module {self.goal_distance_module_id} not in module_ids or sensor_module_ids"
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
        self.enable_dashboard = real_cfg.get("enable_dashboard", True)
        self.dashboard_type = real_cfg.get("dashboard_type", "rl")  # "rl", "motor", "basic"
        self.dashboard_theme = real_cfg.get("dashboard_theme", "cyber")  # cyber, matrix, minimal, retro
        self.dashboard_fullscreen = real_cfg.get("dashboard_fullscreen", True)  # False = compact mode
        self.dashboard_capture_prints = real_cfg.get("dashboard_capture_prints", True)  # Capture print() to log
        self.dashboard = None  # Can be RLDashboard or MotorDashboard
        
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
        self.module_data: Dict[int, SensorData] = {}
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
        print(f"  Active modules: {self.expected_module_ids}")
        print(f"  Sensor modules: {self.sensor_module_ids if self.sensor_module_ids else 'none'}")
        print(f"  Num actions: {len(self.expected_module_ids)}")
        print(f"  Listen port: {self.listen_port}")
        print(f"  Command port: {self.command_port}")
        print(f"  Kp: {self.kp_default}, Kd: {self.kd_default}")
        print(f"  Dashboard: {'enabled' if self.enable_dashboard else 'disabled'}")
        print("-" * 60)
        print("Global State Sources:")
        print(f"  Main IMU (quat/gyro): module {self.main_imu_module_id}")
        if self.goal_distance_module_id is not None:
            print(f"  Goal distance: module {self.goal_distance_module_id}")
        else:
            print(f"  Goal distance: per-module only (no global source)")
        print("=" * 60)
        print("\nAction -> Module mapping:")
        for i, mod_id in enumerate(self.expected_module_ids):
            print(f"  action[{i}] -> module {mod_id}")
        if self.sensor_module_ids:
            print("\nSensor modules (no action):")
            for mod_id in self.sensor_module_ids:
                print(f"  module {mod_id} (sensor only)")
        print("\nWaiting for ESP32 modules to connect...")

    def _init_network_server(self) -> None:
        """Initialize the NetworkServer for ESP32 communication."""
        self.server = NetworkServer(
            recv_type=SensorData,
            send_type=MotorCommand,
            recv_port=self.listen_port,
            send_port=self.command_port,
            callback=self._on_module_feedback,
            timeout_sec=self.device_timeout,
        )

    def _init_dashboard(self) -> None:
        """Initialize the Rich dashboard for real-time monitoring.
        
        Supports two dashboard types:
        - "rl": Enhanced RLDashboard with observation/action visualization
        - "motor": Simple MotorDashboard with motor status only
        """
        if not DASHBOARD_AVAILABLE:
            print("[Warning] Dashboard not available. Install rich: pip install rich")
            self.enable_dashboard = False
            return
        
        try:
            # Try RLDashboard first (enhanced multi-panel layout)
            if self.dashboard_type == "rl" and RL_DASHBOARD_AVAILABLE:
                config = RLDashboardConfig(
                    title="🤖 RealMetaMachine",
                    refresh_rate=20,
                    timeout_sec=self.device_timeout,
                    theme=self.dashboard_theme,
                    show_observations=True,
                    show_actions=True,
                    show_rewards=True,
                    history_steps=self.cfg.observation.get("include_history_steps", 3),
                    fullscreen=self.dashboard_fullscreen,
                    capture_prints=self.dashboard_capture_prints,
                )
                self.dashboard = RLDashboard(config)
                
                # Set expected modules for tracking
                self.dashboard.set_expected_modules(
                    self.expected_module_ids, 
                    self.sensor_module_ids
                )
                
                # Log startup info
                self.dashboard.log_info(f"Listening on port {self.listen_port}")
                self.dashboard.log_info(f"Expecting {len(self.expected_module_ids)} active modules")
                if self.sensor_module_ids:
                    self.dashboard.log_info(f"Expecting {len(self.sensor_module_ids)} sensor modules")
                if not self.dashboard_fullscreen:
                    self.dashboard.log_info("Compact mode - prints visible below")
                
                self.dashboard.start()
            else:
                # Fallback to MotorDashboard
                config = DashboardConfig(
                    title="RealMetaMachine Controller",
                    refresh_rate=20,
                    timeout_sec=self.device_timeout,
                )
                self.dashboard = MotorDashboard(config)
                self.dashboard.start()
                print("[Dashboard] MotorDashboard started")
        except Exception as e:
            print(f"[Warning] Failed to start dashboard: {e}")
            import traceback
            traceback.print_exc()
            self.dashboard = None
            self.enable_dashboard = False

    def _on_module_feedback(self, msg: SensorData, sender_ip: str) -> None:
        """Callback when sensor data is received from an ESP32 module.
        
        Args:
            msg: The SensorData message containing sensor readings
            sender_ip: IP address of the sender
        """
        module_id = msg.module_id
        self.fb_count += 1
        
        # Check if this is an expected module (active or sensor)
        if module_id not in self.all_expected_module_ids:
            # Unexpected module - log warning once
            if module_id not in self.connected_modules:
                self._dashboard_log(f"Unexpected module {module_id} @ {sender_ip}", "warn")
            return
        
        # Track new module connections
        if module_id not in self.connected_modules:
            self.connected_modules.add(module_id)
            self.module_to_ip[module_id] = sender_ip
            self.ip_to_module[sender_ip] = module_id
            
            # Determine module type
            is_sensor = module_id in self.sensor_module_ids
            
            # Notify dashboard about module connection
            if self.dashboard is not None and hasattr(self.dashboard, 'module_connected'):
                self.dashboard.module_connected(module_id, sender_ip, is_sensor)
            
            # Check if all modules are now connected
            if self.all_modules_connected():
                total = len(self.expected_module_ids) + len(self.sensor_module_ids)
                self._dashboard_log(f"All {total} modules ready!", "success")
        
        # Update IP if changed (module moved to different network)
        elif self.module_to_ip.get(module_id) != sender_ip:
            old_ip = self.module_to_ip.get(module_id)
            self.module_to_ip[module_id] = sender_ip
            if old_ip:
                self.ip_to_module.pop(old_ip, None)
            self.ip_to_module[sender_ip] = module_id
            self._dashboard_log(f"Module {module_id} IP: {old_ip} → {sender_ip}", "warn")
        
        # Store latest data
        self.module_data[module_id] = msg
        
        # Update dashboard if enabled
        if self.dashboard is not None:
            self._update_dashboard_motor(module_id, msg, sender_ip)

    def _update_dashboard_motor(self, module_id: int, msg: SensorData, sender_ip: str) -> None:
        """Update dashboard with motor data.
        
        Display format:
            - Active modules: "M{id}->A{idx}" or "M{id}->A{idx}*" (if main_imu)
            - Sensor modules: "S{id}" or "S{id}*" (if main_imu) or "S{id}[D]" (if goal_dist source)
        """
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
        
        # Determine module type and build display name
        is_sensor_module = module_id in self.sensor_module_ids
        is_main_imu = module_id == self.main_imu_module_id
        is_goal_dist_source = module_id == self.goal_distance_module_id
        
        # Build name with markers for special roles
        if is_sensor_module:
            # Sensor module format: S{id} with optional markers
            name = f"S{module_id}"
            if is_main_imu:
                name += "[IMU]"
            if is_goal_dist_source:
                name += "[D]"
        else:
            # Active module format: M{id}->A{idx} with optional markers
            try:
                action_idx = self.expected_module_ids.index(module_id)
            except ValueError:
                action_idx = -1
            name = f"M{module_id}→A{action_idx}"
            if is_main_imu:
                name += "★"  # Marker for main IMU
            if is_goal_dist_source:
                name += "[D]"
        
        # Get goal distance (show in dashboard if >= 0)
        goal_distance = getattr(msg, 'goal_distance', -1.0)
        
        # Determine mode display
        if is_sensor_module:
            mode = "Sensor"
        else:
            mode = "Running" if self.motor_enabled else "Idle"
        
        self.dashboard.update_motor(
            address=sender_ip,
            name=name,
            position=motor.pos if motor else 0.0,
            velocity=motor.vel if motor else 0.0,
            torque=motor.torque if motor else 0.0,
            voltage=motor.voltage if motor else 0.0,
            current=motor.current if motor else 0.0,
            mode=mode,
            switch=self.motor_enabled if not is_sensor_module else False,
            error=error_str,
            distance=goal_distance if goal_distance >= 0 else -1.0,
        )

    def all_modules_connected(self) -> bool:
        """Check if all expected modules (active + sensor) are connected.
        
        Returns:
            bool: True if all expected modules are connected
        """
        return all(mid in self.connected_modules for mid in self.all_expected_module_ids)

    def ready(self) -> bool:
        """Check if the robot system is ready for control.
        
        The system is ready when all expected modules (active + sensor) are 
        connected and actively sending data.
        
        Returns:
            bool: True if all expected modules are active
        """
        # Check all expected modules are connected
        if not self.all_modules_connected():
            return False
        
        # Check all modules are in active devices (recently seen)
        active_ips = set(self.server.active_devices.keys())
        for module_id in self.all_expected_module_ids:
            ip = self.module_to_ip.get(module_id)
            if ip not in active_ips:
                return False
        
        return True

    def get_missing_modules(self) -> List[int]:
        """Get list of expected modules (active + sensor) that are not yet connected.
        
        Returns:
            List of module IDs that are expected but not connected
        """
        return [mid for mid in self.all_expected_module_ids if mid not in self.connected_modules]

    def get_inactive_modules(self) -> List[int]:
        """Get list of modules that are connected but not actively sending.
        
        Returns:
            List of module IDs that are connected but inactive
        """
        active_ips = set(self.server.active_devices.keys())
        inactive = []
        for module_id in self.all_expected_module_ids:
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
            
        Global state sources are determined by the `sources` config:
            - main_imu: Module ID for quat/gyro (default: first module_id)
            - goal_distance: Module ID for global goal_distance (default: None)
        """
        # Process incoming messages
        self.receive_module_data()
        
        num_actions = len(self.expected_module_ids)
        
        # Initialize arrays in action order
        dof_pos = np.zeros(num_actions)
        dof_vel = np.zeros(num_actions)
        
        # Per-module data lists (in action order, for active modules only)
        imu_quats = []
        imu_gyros = []
        imu_accels = []
        goal_distances = []
        
        # Collect data for each active module IN ORDER
        for action_idx, module_id in enumerate(self.expected_module_ids):
            if module_id in self.module_data:
                data = self.module_data[module_id]
                
                # Motor data
                dof_pos[action_idx] = data.motor.pos
                dof_vel[action_idx] = data.motor.vel
                
                # Goal distance (per-module)
                goal_distance = getattr(data, 'goal_distance', 0.0)
                goal_distances.append(goal_distance)
                
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
                
            else:
                # Module not yet received data - use defaults
                imu_quats.append(np.array([0, 0, 0, 1]))
                imu_gyros.append(np.zeros(3))
                imu_accels.append(np.array([0, 0, -9.81]))
                goal_distances.append(0.0)
        
        # =====================================================================
        # Get global state from configured source modules
        # =====================================================================
        
        # Main IMU data (quat, gyro) from configured main_imu module
        main_quat, main_gyro = self._get_module_imu_data(self.main_imu_module_id)
        
        # Global goal distance from configured module (if specified)
        global_goal_distance = None
        if self.goal_distance_module_id is not None:
            global_goal_distance = self._get_module_goal_distance(self.goal_distance_module_id)
        
        self.observable_data = {
            # Joint state (ordered by action index)
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            
            # Global (torso) state - from configured main_imu module
            "quat": main_quat,
            "ang_vel_body": main_gyro,
            "vel_body": np.zeros(3),  # Not available from IMU
            
            # Per-module data (ordered by action index, active modules only)
            "gyros": imu_gyros,
            "quats": imu_quats,
            "accs": imu_accels,
            "goal_distances": goal_distances,  # Per-module goal distances
            
            # Global goal distance (from configured source, if any)
            "goal_distance": global_goal_distance,
            
            # Metadata
            "timestamp": time.time(),
            "module_order": self.expected_module_ids,
            "sensor_modules": self.sensor_module_ids,
            "main_imu_module": self.main_imu_module_id,
            "goal_distance_module": self.goal_distance_module_id,
        }
        
        # Update dashboard performance and RL state
        if self.dashboard is not None:
            self._update_dashboard_performance()
            # Update observation components for RLDashboard
            self._update_dashboard_observations_from_state()
        
        # Log data if enabled
        if self.cfg.logging.get("log_raw_data", False):
            self._log_data()

        return self.observable_data
    
    def _get_module_imu_data(self, module_id: int) -> tuple:
        """Get IMU data (quat, gyro) from a specific module.
        
        Args:
            module_id: The module ID to get IMU data from
            
        Returns:
            tuple: (quat, gyro) arrays
        """
        if module_id in self.module_data:
            data = self.module_data[module_id]
            quat = np.array([
                data.imu.quaternion.x,
                data.imu.quaternion.y,
                data.imu.quaternion.z,
                data.imu.quaternion.w
            ])
            gyro = np.array([
                data.imu.omega.x,
                data.imu.omega.y,
                data.imu.omega.z
            ])
            return quat, gyro
        else:
            # Module not yet received data - use defaults
            return np.array([0, 0, 0, 1]), np.zeros(3)
    
    def _get_module_goal_distance(self, module_id: int) -> float:
        """Get goal_distance from a specific module.
        
        Args:
            module_id: The module ID to get goal_distance from
            
        Returns:
            float: The goal_distance value (0.0 if not available)
        """
        if module_id in self.module_data:
            data = self.module_data[module_id]
            return getattr(data, 'goal_distance', 0.0)
        return 0.0

    def _rotate_vector_by_quat(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        qx, qy, qz, qw = q
        q_vec = np.array([qx, qy, qz])
        cross1 = np.cross(q_vec, v)
        cross2 = np.cross(q_vec, cross1)
        return v + 2.0 * qw * cross1 + 2.0 * cross2

    # =========================================================================
    # Dashboard Update Methods
    # =========================================================================
    
    def _update_dashboard_performance(self) -> None:
        """Update dashboard with performance metrics and status."""
        if self.dashboard is None:
            return
        
        # Common status updates for both dashboard types
        total_expected = len(self.expected_module_ids) + len(self.sensor_module_ids)
        
        # Check if using RLDashboard (has update_performance method)
        if hasattr(self.dashboard, 'update_performance'):
            # RLDashboard - update performance metrics
            self.dashboard.update_performance(
                loop_dt=self.send_dt,
                compute_time=self.compute_time,
                cmd_count=self.cmd_count,
                fb_count=self.fb_count
            )
            self.dashboard.set_status("Modules", f"{len(self.connected_modules)}/{total_expected}")
        else:
            # MotorDashboard - use set_status
            self.dashboard.set_status("Cmd/Fb", f"{self.cmd_count}/{self.fb_count}")
            self.dashboard.set_status("Modules", f"{len(self.connected_modules)}/{total_expected}")
        
        self.dashboard.update()
    
    def _update_dashboard_observation(self, obs_components: Dict[str, Any]) -> None:
        """Update dashboard with observation component data.
        
        Args:
            obs_components: Dict mapping component names to values
        """
        if self.dashboard is None:
            return
        
        # Only RLDashboard supports observation updates
        if hasattr(self.dashboard, 'update_observation'):
            self.dashboard.update_observation(obs_components)
    
    def _update_dashboard_action(self, action: np.ndarray) -> None:
        """Update dashboard with current action.
        
        Args:
            action: Action array being sent to motors
        """
        if self.dashboard is None:
            return
        
        # Only RLDashboard supports action updates
        if hasattr(self.dashboard, 'update_action'):
            self.dashboard.update_action(action)
    
    def _update_dashboard_reward(self, reward: float, episode_reward: float = None) -> None:
        """Update dashboard with reward data.
        
        Args:
            reward: Step reward
            episode_reward: Total episode reward (optional)
        """
        if self.dashboard is None:
            return
        
        if hasattr(self.dashboard, 'update_reward'):
            self.dashboard.update_reward(reward, episode_reward)
    
    def _update_dashboard_step(self) -> None:
        """Increment dashboard step counter."""
        if self.dashboard is None:
            return
        
        if hasattr(self.dashboard, 'increment_step'):
            self.dashboard.increment_step()
    
    def _update_dashboard_new_episode(self) -> None:
        """Signal new episode to dashboard."""
        if self.dashboard is None:
            return
        
        if hasattr(self.dashboard, 'new_episode'):
            self.dashboard.new_episode()
    
    def _dashboard_log(self, message: str, level: str = "info") -> None:
        """Log a message to the dashboard (if RLDashboard).
        
        Args:
            message: Message to log
            level: "info", "warn", "error", "success"
        """
        if self.dashboard is None:
            return
        
        if hasattr(self.dashboard, 'log'):
            self.dashboard.log(message, level)
    
    def _update_dashboard_observations_from_state(self) -> None:
        """Extract and update observation components from the state manager.
        
        This method reads the current observation components from self.state
        and sends them to the RLDashboard for visualization.
        
        Components are split into:
        - Used components (●): Actually used in policy observation
        - Debug components (○): Raw data for debugging
        """
        if self.dashboard is None or not hasattr(self.dashboard, 'update_observation'):
            return
        
        if not hasattr(self, 'state'):
            return
        
        used_components = {}
        debug_components = {}
        
        try:
            # Get observation components actually used in policy
            if hasattr(self.state, 'observation_components') and self.state.observation_components:
                for component in self.state.observation_components:
                    try:
                        data = component.get_data(self.state)
                        if data is not None:
                            used_components[component.name] = np.asarray(data).flatten()
                    except Exception:
                        pass  # Skip components that fail
            
            # Also include raw observable data for debugging
            if self.observable_data:
                # Add key raw values (these are debug data)
                for key in ['dof_pos', 'dof_vel', 'ang_vel_body']:
                    if key in self.observable_data:
                        val = self.observable_data[key]
                        if val is not None and key not in used_components:
                            debug_components[f"raw_{key}"] = np.asarray(val).flatten()
                
                # Add main quat as roll/pitch/yaw for easier visualization
                if 'quat' in self.observable_data:
                    quat = self.observable_data['quat']
                    if quat is not None:
                        rpy = self._quat_to_rpy(quat)
                        debug_components['rpy_deg'] = np.degrees(rpy)
                
                # Add goal distance if available
                if self.observable_data.get('goal_distance') is not None:
                    debug_components['goal_dist'] = np.array([self.observable_data['goal_distance']])
            
            # Add derived state for debugging
            if hasattr(self.state, 'derived'):
                if hasattr(self.state.derived, 'projected_gravity'):
                    pg = self.state.derived.projected_gravity
                    if pg is not None and 'projected_gravity' not in used_components:
                        debug_components['proj_gravity'] = np.asarray(pg).flatten()
                if hasattr(self.state.derived, 'speed'):
                    speed = self.state.derived.speed
                    if speed is not None:
                        debug_components['speed'] = np.asarray(speed).flatten()
            
            # Update dashboard - mark used components appropriately
            if used_components:
                self.dashboard.update_observation(used_components, used_in_policy=True)
            if debug_components:
                self.dashboard.update_observation(debug_components, used_in_policy=False)
                
        except Exception as e:
            # Don't crash on dashboard update failures
            pass
    
    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [x,y,z,w] to roll-pitch-yaw."""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    def _wait_until_motor_on(self) -> None:
        """Wait until all expected robot modules (active + sensor) are ready."""
        wait_count = 0
        rate = Rate(50.0)  # 50 Hz check rate
        
        while not self.ready():
            # Process incoming messages
            self.receive_module_data()
            
            # Send keepalive to connected active modules
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
                total = len(self.all_expected_module_ids)
                
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
        
        # Update dashboard with action (before sending)
        self._update_dashboard_action(pos)
        
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
        
        # Update dashboard step counter
        self._update_dashboard_step()
        
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
            cmd = MotorCommand(
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
        
        # Signal new episode to dashboard
        self._update_dashboard_new_episode()
        
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
            cmd = MotorCommand(
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
            cmd = MotorCommand(
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

    def step(self, action: np.ndarray):
        """Execute one environment step with dashboard updates.
        
        Extends the base step() to add reward tracking for the dashboard.
        """
        # Call parent step
        obs, reward, done, truncated, info = super().step(action)
        
        # Update dashboard with reward info
        if self.dashboard is not None:
            episode_reward = sum(self.episode_rewards) if hasattr(self, 'episode_rewards') else reward
            self._update_dashboard_reward(reward, episode_reward)
        
        return obs, reward, done, truncated, info

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
        active_connected = [m for m in self.connected_modules if m in self.expected_module_ids]
        sensor_connected = [m for m in self.connected_modules if m in self.sensor_module_ids]
        print(f"  Active modules: {active_connected}")
        if self.sensor_module_ids:
            print(f"  Sensor modules: {sensor_connected}")
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
        """Get status of all modules (active and sensor)."""
        status = {}
        
        # Active modules
        for module_id in self.expected_module_ids:
            if module_id in self.module_data:
                data = self.module_data[module_id]
                status[module_id] = {
                    "type": "active",
                    "connected": True,
                    "action_index": self.expected_module_ids.index(module_id),
                    "ip": self.module_to_ip.get(module_id, "unknown"),
                    "pos": data.motor.pos,
                    "vel": data.motor.vel,
                    "torque": data.motor.torque,
                    "temperature": data.motor.temperature,
                    "voltage": data.motor.voltage,
                    "current": data.motor.current,
                    "goal_distance": getattr(data, 'goal_distance', None),
                }
            else:
                status[module_id] = {
                    "type": "active",
                    "connected": False,
                    "action_index": self.expected_module_ids.index(module_id),
                }
        
        # Sensor modules
        for module_id in self.sensor_module_ids:
            if module_id in self.module_data:
                data = self.module_data[module_id]
                status[module_id] = {
                    "type": "sensor",
                    "connected": True,
                    "ip": self.module_to_ip.get(module_id, "unknown"),
                    "goal_distance": getattr(data, 'goal_distance', None),
                }
            else:
                status[module_id] = {
                    "type": "sensor",
                    "connected": False,
                }
        
        return status

    def print_status(self) -> None:
        """Print current status of all modules."""
        print("\n" + "=" * 60)
        print("Module Status")
        print("=" * 60)
        
        # Active modules
        print("\nActive Modules:")
        for i, module_id in enumerate(self.expected_module_ids):
            status = "✓" if module_id in self.connected_modules else "✗"
            ip = self.module_to_ip.get(module_id, "not connected")
            is_main = " [main_imu]" if module_id == self.main_imu_module_id else ""
            is_dist = " [goal_dist]" if module_id == self.goal_distance_module_id else ""
            
            if module_id in self.module_data:
                data = self.module_data[module_id]
                pos = data.motor.pos
                vel = data.motor.vel
                print(f"  [{status}] action[{i}] -> module {module_id} @ {ip}{is_main}{is_dist}")
                print(f"        pos={pos:+.3f}, vel={vel:+.3f}")
            else:
                print(f"  [{status}] action[{i}] -> module {module_id} @ {ip}{is_main}{is_dist}")
        
        # Sensor modules
        if self.sensor_module_ids:
            print("\nSensor Modules:")
            for module_id in self.sensor_module_ids:
                status = "✓" if module_id in self.connected_modules else "✗"
                ip = self.module_to_ip.get(module_id, "not connected")
                is_main = " [main_imu]" if module_id == self.main_imu_module_id else ""
                is_dist = " [goal_dist]" if module_id == self.goal_distance_module_id else ""
                
                if module_id in self.module_data:
                    data = self.module_data[module_id]
                    goal_dist = getattr(data, 'goal_distance', None)
                    print(f"  [{status}] module {module_id} (sensor) @ {ip}{is_main}{is_dist}")
                    if goal_dist is not None:
                        print(f"        goal_distance={goal_dist:.3f}")
                else:
                    print(f"  [{status}] module {module_id} (sensor) @ {ip}{is_main}{is_dist}")
        
        print("=" * 60)
