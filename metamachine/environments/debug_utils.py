"""
Debug utilities for identifying segmentation faults in MetaMachine environments.

This module provides:
1. Signal handlers to catch segmentation faults with stack traces
2. Function decorators to trace execution and identify faulty code
3. Safe wrappers for potentially unsafe operations (rendering, MuJoCo ops)
4. Diagnostic tools to test individual components

Usage:
    # Enable debug mode via environment variable before importing
    export METAMACHINE_DEBUG=1
    
    # Or programmatically:
    from metamachine.environments.debug_utils import enable_debug_mode, DebugEnvironment
    enable_debug_mode()
    
    # Wrap your environment for debugging
    env = DebugEnvironment(cfg)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

import atexit
import faulthandler
import functools
import os
import signal
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

# Debug configuration
_DEBUG_ENABLED = os.environ.get("METAMACHINE_DEBUG", "0") == "1"
_DEBUG_LOG_FILE = None
_DEBUG_VERBOSE = os.environ.get("METAMACHINE_DEBUG_VERBOSE", "0") == "1"
_CURRENT_OPERATION = "initialization"
_OPERATION_STACK = []


def enable_debug_mode(
    log_file: Optional[str] = None,
    verbose: bool = False,
    enable_faulthandler: bool = True,
) -> None:
    """Enable debug mode with comprehensive error tracking.
    
    Args:
        log_file: Path to log file for debug output. If None, uses stderr.
        verbose: Enable verbose output for all operations
        enable_faulthandler: Enable Python faulthandler for native crashes
    """
    global _DEBUG_ENABLED, _DEBUG_LOG_FILE, _DEBUG_VERBOSE
    
    _DEBUG_ENABLED = True
    _DEBUG_VERBOSE = verbose
    
    if log_file:
        _DEBUG_LOG_FILE = open(log_file, "a")
        print(f"[DEBUG] Debug log file: {log_file}")
    
    if enable_faulthandler:
        # Enable faulthandler to print tracebacks on segfaults
        faulthandler.enable(file=_DEBUG_LOG_FILE or sys.stderr, all_threads=True)
        
        # Register signal handlers
        _setup_signal_handlers()
    
    _log_debug("Debug mode enabled")
    _log_debug(f"Python version: {sys.version}")
    _log_debug(f"NumPy version: {np.__version__}")
    
    try:
        import mujoco
        _log_debug(f"MuJoCo version: {mujoco.__version__}")
    except ImportError:
        _log_debug("MuJoCo not installed")
    
    # Register cleanup
    atexit.register(_cleanup_debug)


def _setup_signal_handlers() -> None:
    """Setup signal handlers to catch crashes."""
    def _signal_handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        _log_debug(f"\n{'='*60}")
        _log_debug(f"FATAL: Received signal {sig_name} ({signum})")
        _log_debug(f"Current operation: {_CURRENT_OPERATION}")
        _log_debug(f"Operation stack: {' -> '.join(_OPERATION_STACK)}")
        _log_debug(f"{'='*60}")
        _log_debug("Python traceback:")
        traceback.print_stack(frame, file=_DEBUG_LOG_FILE or sys.stderr)
        _log_debug(f"{'='*60}")
        
        # Dump all threads
        faulthandler.dump_traceback(file=_DEBUG_LOG_FILE or sys.stderr, all_threads=True)
        
        sys.exit(128 + signum)
    
    # Only register handlers for signals we can handle
    try:
        signal.signal(signal.SIGABRT, _signal_handler)
    except (ValueError, OSError):
        pass  # Can't set handler in some contexts
    
    try:
        signal.signal(signal.SIGBUS, _signal_handler)
    except (ValueError, OSError, AttributeError):
        pass  # SIGBUS not available on all platforms


def _cleanup_debug() -> None:
    """Cleanup debug resources."""
    global _DEBUG_LOG_FILE
    if _DEBUG_LOG_FILE:
        _DEBUG_LOG_FILE.close()
        _DEBUG_LOG_FILE = None


def _log_debug(message: str) -> None:
    """Log debug message with timestamp."""
    if not _DEBUG_ENABLED:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_msg = f"[DEBUG {timestamp}] {message}"
    
    output = _DEBUG_LOG_FILE or sys.stderr
    print(log_msg, file=output, flush=True)


def set_current_operation(operation: str) -> None:
    """Set the current operation for crash tracking."""
    global _CURRENT_OPERATION
    _CURRENT_OPERATION = operation
    if _DEBUG_VERBOSE:
        _log_debug(f"Operation: {operation}")


def push_operation(operation: str) -> None:
    """Push operation to stack for nested tracking."""
    global _CURRENT_OPERATION
    _OPERATION_STACK.append(_CURRENT_OPERATION)
    _CURRENT_OPERATION = operation
    if _DEBUG_VERBOSE:
        _log_debug(f">> {operation}")


def pop_operation() -> None:
    """Pop operation from stack."""
    global _CURRENT_OPERATION
    if _OPERATION_STACK:
        _CURRENT_OPERATION = _OPERATION_STACK.pop()
    else:
        _CURRENT_OPERATION = "unknown"
    if _DEBUG_VERBOSE:
        _log_debug(f"<< {_CURRENT_OPERATION}")


def debug_trace(operation_name: str) -> Callable:
    """Decorator to trace function execution for debugging.
    
    Args:
        operation_name: Name to identify this operation in crash logs
        
    Example:
        @debug_trace("video_capture")
        def capture_frame(self):
            # potentially crashing code
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _DEBUG_ENABLED:
                return func(*args, **kwargs)
            
            push_operation(operation_name)
            try:
                if _DEBUG_VERBOSE:
                    _log_debug(f"Entering {operation_name}")
                result = func(*args, **kwargs)
                if _DEBUG_VERBOSE:
                    _log_debug(f"Exiting {operation_name} (success)")
                return result
            except Exception as e:
                _log_debug(f"Exception in {operation_name}: {type(e).__name__}: {e}")
                raise
            finally:
                pop_operation()
        return wrapper
    return decorator


@contextmanager
def debug_section(name: str):
    """Context manager for tracking debug sections.
    
    Example:
        with debug_section("egl_render"):
            renderer.render()
    """
    if not _DEBUG_ENABLED:
        yield
        return
    
    push_operation(name)
    try:
        yield
    except Exception as e:
        _log_debug(f"Exception in section '{name}': {type(e).__name__}: {e}")
        raise
    finally:
        pop_operation()


def safe_mujoco_call(func: Callable, *args, operation_name: str = "mujoco_call", **kwargs) -> Any:
    """Safely execute a MuJoCo function with crash tracking.
    
    Args:
        func: MuJoCo function to call
        *args: Arguments to pass to the function
        operation_name: Name for this operation in crash logs
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    if not _DEBUG_ENABLED:
        return func(*args, **kwargs)
    
    push_operation(f"mujoco:{operation_name}")
    try:
        if _DEBUG_VERBOSE:
            _log_debug(f"MuJoCo call: {func.__name__}")
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        _log_debug(f"MuJoCo error in {operation_name}: {type(e).__name__}: {e}")
        raise
    finally:
        pop_operation()


class DebugMixin:
    """Mixin class to add debugging to MetaMachine environment.
    
    Add this as a mixin to your environment class:
    
        class DebuggableMetaMachine(DebugMixin, MetaMachine):
            pass
    """
    
    def _debug_init(self) -> None:
        """Initialize debug tracking."""
        self._debug_step_count = 0
        self._debug_render_count = 0
        self._debug_reset_count = 0
    
    @debug_trace("env_reset")
    def reset(self, *args, **kwargs):
        """Reset with debug tracking."""
        self._debug_reset_count = getattr(self, "_debug_reset_count", 0) + 1
        _log_debug(f"Reset #{self._debug_reset_count}")
        return super().reset(*args, **kwargs)
    
    @debug_trace("env_step")
    def step(self, action):
        """Step with debug tracking."""
        self._debug_step_count = getattr(self, "_debug_step_count", 0) + 1
        if _DEBUG_VERBOSE or self._debug_step_count % 100 == 0:
            _log_debug(f"Step #{self._debug_step_count}")
        return super().step(action)
    
    @debug_trace("env_render")
    def render(self, *args, **kwargs):
        """Render with debug tracking."""
        self._debug_render_count = getattr(self, "_debug_render_count", 0) + 1
        if _DEBUG_VERBOSE or self._debug_render_count % 100 == 0:
            _log_debug(f"Render #{self._debug_render_count}")
        return super().render(*args, **kwargs)
    
    @debug_trace("create_egl_renderer")
    def _create_egl_renderer(self):
        """Create EGL renderer with debug tracking."""
        return super()._create_egl_renderer()
    
    @debug_trace("capture_frame_egl")
    def _capture_frame_egl(self):
        """Capture frame with debug tracking."""
        return super()._capture_frame_egl()
    
    @debug_trace("cleanup_egl_renderer")
    def _cleanup_egl_renderer(self):
        """Cleanup renderer with debug tracking."""
        return super()._cleanup_egl_renderer()
    
    @debug_trace("save_video")
    def _save_video(self):
        """Save video with debug tracking."""
        return super()._save_video()
    
    @debug_trace("reload_model")
    def reload_model(self, xml_string: str):
        """Reload model with debug tracking."""
        return super().reload_model(xml_string)
    
    @debug_trace("do_simulation")
    def do_simulation(self, ctrl, n_frames):
        """Do simulation with debug tracking."""
        return super().do_simulation(ctrl, n_frames)


def create_debug_environment(cfg):
    """Create a MetaMachine environment with debug instrumentation.
    
    Args:
        cfg: OmegaConf configuration for the environment
        
    Returns:
        Instrumented MetaMachine environment
    """
    from metamachine.environments.env_sim import MetaMachine
    
    # Create instrumented class
    class DebugMetaMachine(DebugMixin, MetaMachine):
        def __init__(self, cfg):
            self._debug_init()
            super().__init__(cfg)
    
    enable_debug_mode(verbose=True)
    return DebugMetaMachine(cfg)


# Component test functions for isolating issues

def test_mujoco_basic() -> bool:
    """Test basic MuJoCo functionality."""
    _log_debug("Testing basic MuJoCo...")
    try:
        import mujoco
        
        with debug_section("mujoco_model_create"):
            xml = """
            <mujoco>
                <worldbody>
                    <body>
                        <joint type="free"/>
                        <geom type="sphere" size="0.1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
        
        with debug_section("mujoco_step"):
            for _ in range(10):
                mujoco.mj_step(model, data)
        
        _log_debug("MuJoCo basic test: PASSED")
        return True
    except Exception as e:
        _log_debug(f"MuJoCo basic test: FAILED - {e}")
        return False


def test_mujoco_renderer() -> bool:
    """Test MuJoCo renderer (potential segfault source)."""
    _log_debug("Testing MuJoCo renderer...")
    try:
        import mujoco
        
        xml = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="free"/>
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        with debug_section("renderer_create"):
            renderer = mujoco.Renderer(model, height=240, width=426)
        
        with debug_section("renderer_update_scene"):
            renderer.update_scene(data)
        
        with debug_section("renderer_render"):
            pixels = renderer.render()
            _log_debug(f"Rendered frame shape: {pixels.shape}")
        
        with debug_section("renderer_close"):
            renderer.close()
        
        _log_debug("MuJoCo renderer test: PASSED")
        return True
    except Exception as e:
        _log_debug(f"MuJoCo renderer test: FAILED - {e}")
        return False


def test_egl_context() -> bool:
    """Test EGL context creation."""
    _log_debug("Testing EGL context...")
    try:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        
        # Try to create an EGL context through MuJoCo
        return test_mujoco_renderer()
    except Exception as e:
        _log_debug(f"EGL context test: FAILED - {e}")
        return False


def test_video_recording() -> bool:
    """Test video recording functionality."""
    _log_debug("Testing video recording...")
    try:
        import tempfile
        
        import cv2
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        
        frames = []
        for i in range(10):
            with debug_section(f"create_frame_{i}"):
                frame = np.random.randint(0, 255, (240, 426, 3), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            video_path = f.name
            
            with debug_section("create_video_clip"):
                clip = ImageSequenceClip(frames, fps=20)
            
            with debug_section("write_video"):
                clip.write_videofile(
                    video_path,
                    codec="libx264",
                    fps=20,
                    audio=False,
                    logger=None,
                )
            
            with debug_section("close_clip"):
                clip.close()
        
        _log_debug("Video recording test: PASSED")
        return True
    except Exception as e:
        _log_debug(f"Video recording test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gymnasium_mujoco_renderer() -> bool:
    """Test gymnasium's MuJoCo renderer."""
    _log_debug("Testing gymnasium MuJoCo renderer...")
    try:
        import inspect
        
        import mujoco
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        
        xml = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="free"/>
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        model.vis.global_.offwidth = 426
        model.vis.global_.offheight = 240
        data = mujoco.MjData(model)
        
        with debug_section("gymnasium_renderer_create"):
            # Check the signature to handle different gymnasium versions
            sig = inspect.signature(MujocoRenderer.__init__)
            params = list(sig.parameters.keys())
            
            if "default_camera_config" in params:
                # Older gymnasium version
                renderer = MujocoRenderer(
                    model,
                    data,
                    default_camera_config={"distance": 4.0},
                    width=426,
                    height=240,
                )
            else:
                # Newer gymnasium version - requires width/height explicitly
                kwargs = {"model": model, "data": data}
                if "width" in params:
                    kwargs["width"] = 426
                if "height" in params:
                    kwargs["height"] = 240
                renderer = MujocoRenderer(**kwargs)
        
        with debug_section("gymnasium_renderer_render"):
            frame = renderer.render("rgb_array")
            _log_debug(f"Rendered frame shape: {frame.shape if frame is not None else 'None'}")
        
        with debug_section("gymnasium_renderer_close"):
            renderer.close()
        
        _log_debug("Gymnasium MuJoCo renderer test: PASSED")
        return True
    except Exception as e:
        _log_debug(f"Gymnasium MuJoCo renderer test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_diagnostic_tests() -> dict:
    """Run all diagnostic tests and return results."""
    enable_debug_mode(verbose=True)
    
    _log_debug("=" * 60)
    _log_debug("Running MetaMachine diagnostic tests")
    _log_debug("=" * 60)
    
    results = {
        "mujoco_basic": test_mujoco_basic(),
        "mujoco_renderer": test_mujoco_renderer(),
        "egl_context": test_egl_context(),
        "video_recording": test_video_recording(),
        "gymnasium_renderer": test_gymnasium_mujoco_renderer(),
    }
    
    _log_debug("=" * 60)
    _log_debug("Diagnostic test results:")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        _log_debug(f"  {test_name}: {status}")
    _log_debug("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run diagnostic tests when module is executed directly
    results = run_diagnostic_tests()
    
    # Exit with error code if any test failed
    if not all(results.values()):
        sys.exit(1)

