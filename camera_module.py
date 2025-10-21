"""
camera_module.py - Unified camera module with singleton pattern and frame distribution

This module provides a single-stream, multi-consumer camera system for applications
that need to share camera frames between multiple components (GUI, processing, recording, etc.)
without the overhead and complexity of multiple video streams.

Architecture:
    - CameraHardware: Manages physical camera operations
    - FrameDistributor: Handles frame distribution to multiple consumers
    - CameraStatistics: Tracks performance metrics and FPS
    - CameraManager: Singleton orchestrator providing public interface
"""

import cv2
import logging
import time
import threading
import numpy as np
from typing import Optional, Callable, List, Dict, Any, Tuple
from datetime import datetime
from collections import deque
from error_module import CameraError
from enhanced_config_manager import ModuleConfig

# Initialize module-specific logger
logger = logging.getLogger(__name__)


# ===== CONSUMER SYSTEM =====

class FrameConsumer:
    """
    Represents a consumer that receives camera frames with configurable processing behavior.

    Consumers can be synchronous (blocking) for critical processing or asynchronous
    (non-blocking) for display and logging. Priority determines processing order.
    """

    def __init__(self, name: str, callback: Callable[[np.ndarray], None],
                 processing_type: str = "async", priority: int = 0):
        """
        Initialize a frame consumer.

        Args:
            name: Unique identifier for this consumer
            callback: Function to call with each frame
            processing_type: "async" for non-blocking, "sync" for blocking
            priority: Higher priority consumers get frames first
        """
        self.name = name
        self.callback = callback
        self.processing_type = processing_type
        self.priority = priority
        self.frame_count = 0
        self.last_frame_time = None
        self.is_active = True


# ===== HARDWARE LAYER =====

class CameraHardware:
    """
    Manages physical camera operations including initialization, configuration,
    and frame capture. Handles the low-level OpenCV camera interface.
    """

    def __init__(self, config_manager=None):
        """
        Initialize camera hardware configuration.

        Args:
            config_manager: Configuration manager for camera settings
        """
        # Load configuration
        if config_manager:
            camera_config = config_manager.get_module_config(ModuleConfig.CAMERA.value)
            self.device_id = camera_config.get("device_id", 0)
            self.buffer_size = camera_config.get("buffer_size", 1)
            self.width = camera_config.get("width", 1920)
            self.height = camera_config.get("height", 1080)
            self.fps = camera_config.get("fps", 30)
            self.auto_exposure = camera_config.get("auto_exposure", True)
        else:
            # Default configuration
            self.device_id = 0
            self.buffer_size = 1
            self.width = 1920
            self.height = 1080
            self.fps = 30
            self.auto_exposure = True

        # Hardware state
        self.cap = None
        self.is_initialized = False
        self.is_warmed_up = False
        self.hardware_lock = threading.RLock()

    def initialize(self) -> bool:
        """
        Initialize the camera hardware with configured settings.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        with self.hardware_lock:
            if self.is_initialized:
                logger.info("Camera hardware already initialized")
                return True

            try:
                # Create VideoCapture object
                self.cap = cv2.VideoCapture(self.device_id)

                if not self.cap.isOpened():
                    raise CameraError(f"Failed to open camera {self.device_id}")

                # Configure camera properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                if not self.auto_exposure:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

                # Store actual camera settings
                self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                logger.info(f"Camera initialized: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")

                self.is_initialized = True
                self._warm_up_camera()
                return True

            except Exception as e:
                logger.error(f"Camera initialization failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False

    def _warm_up_camera(self, num_frames: int = 10, delay: float = 0.1):
        """
        Warm up the camera by capturing and discarding initial frames.
        This ensures stable exposure and white balance.

        Args:
            num_frames: Number of frames to discard
            delay: Delay between frames in seconds
        """
        if not self.is_initialized:
            return

        logger.info(f"Warming up camera with {num_frames} frames")

        for i in range(num_frames):
            ret, _ = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame during warm-up ({i + 1}/{num_frames})")
            time.sleep(delay)

        self.is_warmed_up = True
        logger.info("Camera warm-up complete")

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.

        Returns:
            Tuple of (success, frame) where success is bool and frame is np.ndarray or None
        """
        if not self.is_initialized:
            return False, None

        try:
            return self.cap.read()
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return False, None

    def switch_device(self, new_device_id: int) -> bool:
        """
        Switch to a different camera device.

        Args:
            new_device_id: The new camera device ID

        Returns:
            bool: True if switch successful
        """
        logger.info(f"Switching camera from device {self.device_id} to {new_device_id}")

        if self.device_id == new_device_id and self.is_initialized:
            logger.info("Device ID unchanged, skipping switch")
            return True

        # Release current camera
        self.release()

        # Update device ID and reinitialize
        self.device_id = new_device_id
        return self.initialize()

    def release(self):
        """Release camera hardware resources."""
        with self.hardware_lock:
            if self.cap and self.is_initialized:
                self.cap.release()
                self.cap = None
                self.is_initialized = False
                self.is_warmed_up = False
                logger.info("Camera hardware released")

    @staticmethod
    def enumerate_cameras(max_check: int = 3) -> List[Tuple[int, str]]:
        """
        Detect available camera devices by testing each device index.

        Args:
            max_check: Maximum number of camera indices to check. Defaulted to 3 maximum cameras.

        Returns:
            List of tuples (device_id, description)
        """
        available_cameras = []

        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    backend = cap.getBackendName()
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    description = f"Camera {i} ({backend}) - {width}x{height}"
                    available_cameras.append((i, description))
                    logger.info(f"Found camera: {description}")
                cap.release()

        if not available_cameras:
            logger.warning("No cameras found, adding default option")
            available_cameras.append((0, "Default Camera (0)"))

        return available_cameras


# ===== DISTRIBUTION LAYER =====

class FrameDistributor:
    """
    Manages frame distribution to multiple consumers with priority-based ordering
    and configurable processing modes (sync/async).
    """

    def __init__(self):
        """Initialize the frame distributor."""
        self.consumers: Dict[str, FrameConsumer] = {}
        self.consumer_lock = threading.RLock()

    def register_consumer(self, name: str, callback: Callable[[np.ndarray], None],
                          processing_type: str = "async", priority: int = 0) -> bool:
        """
        Register a consumer to receive frames.

        Args:
            name: Unique identifier for the consumer
            callback: Function to call with each frame
            processing_type: "sync" for blocking, "async" for non-blocking
            priority: Higher priority consumers get frames first

        Returns:
            bool: True if registered successfully
        """
        with self.consumer_lock:
            if name in self.consumers:
                logger.warning(f"Consumer '{name}' already registered")
                return False

            consumer = FrameConsumer(name, callback, processing_type, priority)
            self.consumers[name] = consumer

            logger.info(f"Registered consumer '{name}' (type: {processing_type}, priority: {priority})")
            return True

    def unregister_consumer(self, name: str) -> bool:
        """
        Remove a consumer from the distribution list.

        Args:
            name: Name of the consumer to unregister

        Returns:
            bool: True if unregistered successfully
        """
        with self.consumer_lock:
            if name not in self.consumers:
                logger.warning(f"Consumer '{name}' not found")
                return False

            del self.consumers[name]
            logger.info(f"Unregistered consumer '{name}'")
            return True

    def pause_consumer(self, name: str):
        """Temporarily pause a consumer from receiving frames."""
        with self.consumer_lock:
            if name in self.consumers:
                self.consumers[name].is_active = False
                logger.info(f"Paused consumer '{name}'")

    def resume_consumer(self, name: str):
        """Resume a paused consumer."""
        with self.consumer_lock:
            if name in self.consumers:
                self.consumers[name].is_active = True
                logger.info(f"Resumed consumer '{name}'")

    def distribute_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Distribute a captured frame to all registered consumers based on priority.

        Args:
            frame: The captured frame to distribute

        Returns:
            Dict of consumer names to processing latencies
        """
        latencies = {}

        with self.consumer_lock:
            if not self.consumers:
                return latencies

            # Sort consumers by priority (higher priority first)
            sorted_consumers = sorted(
                self.consumers.values(),
                key=lambda c: c.priority,
                reverse=True
            )

            for consumer in sorted_consumers:
                if not consumer.is_active:
                    continue

                try:
                    start_time = time.time()

                    if consumer.processing_type == "sync":
                        # Synchronous processing - blocks until complete
                        consumer.callback(frame.copy())
                    else:
                        # Asynchronous processing - non-blocking
                        threading.Thread(
                            target=consumer.callback,
                            args=(frame.copy(),),
                            daemon=True
                        ).start()

                    # Track performance
                    latencies[consumer.name] = time.time() - start_time
                    consumer.frame_count += 1
                    consumer.last_frame_time = time.time()

                except Exception as e:
                    logger.error(f"Error in consumer '{consumer.name}': {e}")

        return latencies

    def get_consumer_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered consumers.

        Returns:
            Dictionary of consumer information
        """
        with self.consumer_lock:
            return {
                name: {
                    "frame_count": consumer.frame_count,
                    "last_frame_time": consumer.last_frame_time,
                    "is_active": consumer.is_active,
                    "processing_type": consumer.processing_type,
                    "priority": consumer.priority
                }
                for name, consumer in self.consumers.items()
            }

    def clear_consumers(self):
        """Remove all registered consumers."""
        with self.consumer_lock:
            self.consumers.clear()
            logger.info("All consumers cleared")


# ===== STATISTICS LAYER =====

class CameraStatistics:
    """
    Tracks camera performance metrics including FPS, frame counts, and consumer latencies.
    """

    def __init__(self):
        """Initialize statistics tracking."""
        self.stats = {
            "total_frames": 0,
            "dropped_frames": 0,
            "fps_actual": 0,
            "last_frame_time": None,
            "consumer_latencies": {}
        }

        # FPS calculation using rolling window
        self.fps_calc_frames = deque(maxlen=30)
        self.fps_calc_times = deque(maxlen=30)
        self.stats_lock = threading.Lock()

    def update_frame_stats(self, frame_captured: bool, consumer_latencies: Dict[str, float]):
        """
        Update statistics for a frame processing cycle.

        Args:
            frame_captured: Whether frame capture was successful
            consumer_latencies: Dictionary of consumer processing times
        """
        with self.stats_lock:
            current_time = time.time()

            if frame_captured:
                self.stats["total_frames"] += 1
                self.stats["last_frame_time"] = current_time

                # Update FPS calculation
                self.fps_calc_frames.append(self.stats["total_frames"])
                self.fps_calc_times.append(current_time)

                if len(self.fps_calc_times) > 1:
                    time_diff = self.fps_calc_times[-1] - self.fps_calc_times[0]
                    frame_diff = self.fps_calc_frames[-1] - self.fps_calc_frames[0]
                    if time_diff > 0:
                        self.stats["fps_actual"] = frame_diff / time_diff
            else:
                self.stats["dropped_frames"] += 1

            # Update consumer latencies
            self.stats["consumer_latencies"].update(consumer_latencies)

    def get_statistics(self, hardware_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics including camera and consumer metrics.

        Args:
            hardware_info: Optional hardware information to include

        Returns:
            Dictionary containing all performance metrics
        """
        with self.stats_lock:
            camera_stats = {
                "fps_actual": self.stats["fps_actual"],
                "total_frames": self.stats["total_frames"],
                "dropped_frames": self.stats["dropped_frames"],
                "last_frame_time": self.stats["last_frame_time"]
            }

            if hardware_info:
                camera_stats.update(hardware_info)

            return {
                "camera": camera_stats,
                "consumer_latencies": self.stats["consumer_latencies"].copy()
            }

    def reset_statistics(self):
        """Reset all statistics to initial values."""
        with self.stats_lock:
            self.stats = {
                "total_frames": 0,
                "dropped_frames": 0,
                "fps_actual": 0,
                "last_frame_time": None,
                "consumer_latencies": {}
            }
            self.fps_calc_frames.clear()
            self.fps_calc_times.clear()
            logger.info("Statistics reset")


# ===== PUBLIC INTERFACE =====

class CameraManager:
    """
    Singleton camera manager that orchestrates hardware, distribution, and statistics.
    Provides the main public interface for camera operations.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_manager=None):
        """Implement singleton pattern to ensure only one camera instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_manager=None):
        """
        Initialize the camera manager (only runs once due to singleton).

        Args:
            config_manager: Configuration manager for camera settings
        """
        if self._initialized:
            return

        self._initialized = True

        # Initialize components
        self.hardware = CameraHardware(config_manager)
        self.distributor = FrameDistributor()
        self.statistics = CameraStatistics()

        # Capture thread management
        self.capture_thread = None
        self.is_capturing = False
        self.capture_lock = threading.RLock()

        logger.info("Camera manager singleton created")

    def initialize(self) -> bool:
        """
        Initialize camera hardware.

        Returns:
            bool: True if successful, False otherwise
        """
        return self.hardware.initialize()

    def start_capture(self) -> bool:
        """
        Start the continuous frame capture and distribution process.

        Returns:
            bool: True if started successfully
        """
        with self.capture_lock:
            if not self.hardware.is_initialized:
                if not self.initialize():
                    return False

            if self.is_capturing:
                logger.info("Capture already running")
                return True

            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            logger.info("Camera capture started")
            return True

    def stop_capture(self):
        """Stop the continuous capture process."""
        with self.capture_lock:
            if not self.is_capturing:
                return

            self.is_capturing = False

            if self.capture_thread:
                self.capture_thread.join(timeout=2.0)
                if self.capture_thread.is_alive():
                    logger.warning("Capture thread did not stop gracefully")

            logger.info("Camera capture stopped")

    def _capture_loop(self):
        """
        Main capture loop that continuously captures frames and distributes them.
        Runs in a separate daemon thread.
        """
        logger.info("Capture loop started")

        while self.is_capturing:
            try:
                # Capture frame from hardware
                ret, frame = self.hardware.capture_frame()

                if ret and frame is not None:
                    # Distribute frame to all consumers
                    consumer_latencies = self.distributor.distribute_frame(frame)
                    # Update statistics
                    self.statistics.update_frame_stats(True, consumer_latencies)
                else:
                    # Update statistics for dropped frame
                    self.statistics.update_frame_stats(False, {})

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.statistics.update_frame_stats(False, {})

        logger.info("Capture loop ended")

    def register_consumer(self, name: str, callback: Callable[[np.ndarray], None],
                          processing_type: str = "async", priority: int = 0) -> bool:
        """
        Register a consumer to receive camera frames.
        A facade layer so the program as a while only needed to interact with the cameraManager.

        Args:
            name: Unique identifier for the consumer
            callback: Function to call with each frame
            processing_type: "sync" for blocking, "async" for non-blocking
            priority: Higher priority consumers get frames first

        Returns:
            bool: True if registered successfully
        """
        return self.distributor.register_consumer(name, callback, processing_type, priority)

    def unregister_consumer(self, name: str) -> bool:
        """Remove a consumer from receiving frames.
        A facade layer so the program as a while only needed to interact with the cameraManager."""
        return self.distributor.unregister_consumer(name)

    def pause_consumer(self, name: str):
        """Temporarily pause a consumer.
        A facade layer so the program as a while only needed to interact with the cameraManager."""
        self.distributor.pause_consumer(name)

    def resume_consumer(self, name: str):
        """Resume a paused consumer.
        A facade layer so the program as a while only needed to interact with the cameraManager."""
        self.distributor.resume_consumer(name)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a single frame directly (legacy compatibility method).

        Note: This bypasses the distribution system and should only be used
        for compatibility with existing code.

        Returns:
            Captured frame or None if not available
        """
        if not self.hardware.is_initialized:
            return None

        ret, frame = self.hardware.capture_frame()
        if ret:
            self.statistics.update_frame_stats(True, {})
            return frame
        else:
            self.statistics.update_frame_stats(False, {})
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive camera and consumer statistics.

        Returns:
            Dictionary containing performance metrics
        """
        # Gather hardware info
        hardware_info = {
            "device_id": self.hardware.device_id,
            "resolution": f"{getattr(self.hardware, 'actual_width', 'unknown')}x{getattr(self.hardware, 'actual_height', 'unknown')}",
            "fps_requested": self.hardware.fps,
            "is_capturing": self.is_capturing,
            "is_initialized": self.hardware.is_initialized
        }

        # Get statistics with hardware info
        stats = self.statistics.get_statistics(hardware_info)

        # Add consumer information
        stats["consumers"] = self.distributor.get_consumer_info()

        return stats

    def switch_camera(self, new_device_id: int, config_manager=None) -> bool:
        """
        Switch to a different camera device.

        Args:
            new_device_id: The new camera device ID
            config_manager: Optional config manager to update settings

        Returns:
            bool: True if switch was successful
        """
        # Store capture state
        was_capturing = self.is_capturing

        # Stop capture if running
        if was_capturing:
            self.stop_capture()

        # Update config if manager provided
        if config_manager:
            camera_config = config_manager.get_module_config("camera")
            camera_config["device_id"] = new_device_id
            config_manager.update_module_config("camera", camera_config)

        # Switch hardware device
        success = self.hardware.switch_device(new_device_id)

        # Restart capture if it was running
        if success and was_capturing:
            self.start_capture()

        return success

    def release(self):
        """Release all camera resources and cleanup."""
        with self.capture_lock:
            # Stop capture thread
            self.stop_capture()

            # Release hardware
            self.hardware.release()

            # Clear consumers
            self.distributor.clear_consumers()

            # Reset statistics
            self.statistics.reset_statistics()

            logger.info("Camera manager resources released")

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.release()
            cls._instance = None

    @staticmethod
    def enumerate_cameras(max_check: int = 10) -> List[Tuple[int, str]]:
        """
        Detect available camera devices.

        Args:
            max_check: Maximum number of camera indices to check

        Returns:
            List of tuples (device_id, description)
        """
        return CameraHardware.enumerate_cameras(max_check)


# ===== FACTORY FUNCTION =====

def create_camera(camera_type="webcam", config_manager=None):
    """
    Factory function to create/get camera manager instance.

    Args:
        camera_type: Type of camera (currently only "webcam")
        config_manager: Configuration manager instance

    Returns:
        CameraManager: The singleton camera instance
    """
    if camera_type != "webcam":
        raise CameraError(f"Unsupported camera type: {camera_type}")

    camera = CameraManager(config_manager)
    if not camera.hardware.is_initialized:
        if not camera.initialize():
            raise CameraError("Failed to initialize camera")

    return camera


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Create or get the singleton camera instance
    # This initializes hardware and prepares for capture
    camera = create_camera()

    # Register a consumer that will receive every captured frame
    # The lambda function gets called with each frame (async, non-blocking)
    camera.register_consumer("demo", lambda frame: print(f"Frame: {frame.shape}"))

    # Start the background capture thread
    # This begins continuous frame capture and distribution
    camera.start_capture()

    # Let the system run for 2 seconds
    # During this time, frames are captured and your lambda is called
    time.sleep(2)

    # Stop capture and release all resources
    camera.release()