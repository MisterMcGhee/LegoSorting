"""
camera_module.py - Unified camera module with singleton pattern and frame distribution

This module combines:
1. Original camera functionality (initialization, warm-up, frame capture)
2. Singleton resource management (prevents multiple instances)
3. Frame distribution system (single stream, multiple consumers)
4. Thread-safe operations for concurrent access
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


class FrameConsumer:
    """Represents a consumer of camera frames"""

    def __init__(self, name: str, callback: Callable[[np.ndarray], None],
                 processing_type: str = "async", priority: int = 0):
        """
        Initialize a frame consumer

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


class CameraModule:
    """
    Unified camera module with singleton pattern and frame distribution.

    This module handles:
    1. Camera hardware initialization and management
    2. Single video stream capture
    3. Distribution of frames to multiple consumers (GUI, sorting, recording, etc.)
    4. Thread-safe resource management
    5. Performance optimization through frame sharing
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_manager=None):
        """Implement singleton pattern to ensure only one camera instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_manager=None):
        """
        Initialize the camera module (only runs once due to singleton).

        Args:
            config_manager: Configuration manager for camera settings
        """
        # Skip initialization if already done
        if self._initialized:
            return

        self._initialized = True

        # Get camera configuration
        if config_manager:
            camera_config = config_manager.get_module_config(ModuleConfig.CAMERA.value)
            self.device_id = camera_config.get("device_id", 0)
            self.buffer_size = camera_config.get("buffer_size", 1)
            self.width = camera_config.get("width", 1920)
            self.height = camera_config.get("height", 1080)
            self.fps = camera_config.get("fps", 30)
            self.auto_exposure = camera_config.get("auto_exposure", True)
        else:
            # Default settings
            self.device_id = 0
            self.buffer_size = 1
            self.width = 1920
            self.height = 1080
            self.fps = 30
            self.auto_exposure = True

        # Camera handle
        self.cap = None
        self.is_initialized = False
        self.is_warmed_up = False

        # Thread management
        self.capture_thread = None
        self.is_capturing = False
        self.capture_lock = threading.RLock()

        # Frame consumers (GUI display, sorting algorithm, etc.)
        self.consumers: Dict[str, FrameConsumer] = {}
        self.consumer_lock = threading.RLock()

        # Frame buffer for async consumers
        self.frame_buffer = deque(maxlen=5)
        self.buffer_lock = threading.Lock()

        # Performance tracking
        self.stats = {
            "total_frames": 0,
            "dropped_frames": 0,
            "fps_actual": 0,
            "last_frame_time": None,
            "consumer_latencies": {}
        }

        # FPS calculation
        self.fps_calc_frames = deque(maxlen=30)
        self.fps_calc_times = deque(maxlen=30)

        logger.info("Camera module singleton created")

    def initialize(self) -> bool:
        """
        Initialize the camera hardware.

        Returns:
            bool: True if successful, False otherwise
        """
        with self.capture_lock:
            if self.is_initialized:
                logger.info("Camera already initialized")
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
                    # Disable auto exposure for consistent lighting
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust as needed

                # Get actual settings (camera might not support requested values)
                self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                logger.info(f"Camera initialized: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")

                self.is_initialized = True

                # Warm up camera
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

    def start_capture(self) -> bool:
        """
        Start the capture thread that continuously reads frames.

        Returns:
            bool: True if started successfully
        """
        with self.capture_lock:
            if not self.is_initialized:
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
        """Stop the capture thread."""
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
        Main capture loop that runs in a separate thread.

        This method continuously captures frames and distributes them to all
        registered consumers, ensuring efficient single-stream processing.
        """
        logger.info("Capture loop started")

        while self.is_capturing:
            try:
                # Capture frame once
                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to capture frame")
                    self.stats["dropped_frames"] += 1
                    continue

                # Update statistics
                current_time = time.time()
                self.stats["total_frames"] += 1
                self.stats["last_frame_time"] = current_time

                # Calculate FPS
                self.fps_calc_frames.append(self.stats["total_frames"])
                self.fps_calc_times.append(current_time)
                if len(self.fps_calc_times) > 1:
                    time_diff = self.fps_calc_times[-1] - self.fps_calc_times[0]
                    frame_diff = self.fps_calc_frames[-1] - self.fps_calc_frames[0]
                    if time_diff > 0:
                        self.stats["fps_actual"] = frame_diff / time_diff

                # Distribute frame to all consumers
                self._distribute_frame(frame)

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.stats["dropped_frames"] += 1

        logger.info("Capture loop ended")

    def _distribute_frame(self, frame: np.ndarray):
        """
        Distribute a captured frame to all registered consumers.

        This is where the magic happens - a single captured frame is sent
        to multiple consumers (GUI display, sorting algorithm, recorder, etc.)
        without duplicating the capture process.

        Args:
            frame: The captured frame to distribute
        """
        with self.consumer_lock:
            if not self.consumers:
                return

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
                        # Good for critical processing like sorting decisions
                        consumer.callback(frame.copy())
                    else:
                        # Asynchronous processing - non-blocking
                        # Good for display, recording, statistics
                        threading.Thread(
                            target=consumer.callback,
                            args=(frame.copy(),),
                            daemon=True
                        ).start()

                    # Track performance
                    latency = time.time() - start_time
                    self.stats["consumer_latencies"][consumer.name] = latency

                    consumer.frame_count += 1
                    consumer.last_frame_time = time.time()

                except Exception as e:
                    logger.error(f"Error in consumer '{consumer.name}': {e}")

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
        Unregister a consumer.

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

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a single frame directly (legacy compatibility).

        This method is kept for backward compatibility with code that
        expects to poll for frames rather than receive callbacks.

        Returns:
            Captured frame or None if not available
        """
        if not self.is_initialized:
            return None

        if not self.is_warmed_up:
            self._warm_up_camera()

        try:
            ret, frame = self.cap.read()
            if ret:
                self.stats["total_frames"] += 1
                return frame
            else:
                self.stats["dropped_frames"] += 1
                return None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get camera and consumer statistics.

        Returns:
            Dictionary containing performance metrics
        """
        with self.consumer_lock:
            consumer_stats = {
                name: {
                    "frame_count": consumer.frame_count,
                    "last_frame_time": consumer.last_frame_time,
                    "is_active": consumer.is_active,
                    "latency": self.stats["consumer_latencies"].get(name, 0)
                }
                for name, consumer in self.consumers.items()
            }

        return {
            "camera": {
                "device_id": self.device_id,
                "resolution": f"{self.actual_width}x{self.actual_height}",
                "fps_requested": self.fps,
                "fps_actual": self.stats["fps_actual"],
                "total_frames": self.stats["total_frames"],
                "dropped_frames": self.stats["dropped_frames"],
                "is_capturing": self.is_capturing
            },
            "consumers": consumer_stats
        }

    def release(self):
        """Release camera resources."""
        with self.capture_lock:
            # Stop capture thread
            self.stop_capture()

            # Release camera
            if self.cap and self.is_initialized:
                self.cap.release()
                self.cap = None
                self.is_initialized = False
                self.is_warmed_up = False
                logger.info("Camera resources released")

            # Clear consumers
            with self.consumer_lock:
                self.consumers.clear()

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.release()
            cls._instance = None


# Factory function for backward compatibility
def create_camera(camera_type="webcam", config_manager=None):
    """
    Factory function to create/get camera instance.

    Args:
        camera_type: Type of camera (currently only "webcam")
        config_manager: Configuration manager instance

    Returns:
        CameraModule: The singleton camera instance
    """
    if camera_type != "webcam":
        raise CameraError(f"Unsupported camera type: {camera_type}")

    camera = CameraModule(config_manager)
    if not camera.is_initialized:
        if not camera.initialize():
            raise CameraError("Failed to initialize camera")

    return camera


# Example usage showing how different parts of the application use the same stream
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create camera instance
    camera = create_camera()


    # Example consumer callbacks
    def gui_display_callback(frame):
        """GUI display consumer - shows frame in window"""
        # This would update your PyQt display
        print(f"GUI received frame: {frame.shape}")
        # cv2.imshow("GUI Display", frame)


    def sorting_algorithm_callback(frame):
        """Sorting algorithm consumer - processes frame for detection"""
        # This would run your piece detection
        print(f"Sorting algorithm processing frame: {frame.shape}")
        # pieces = detect_pieces(frame)
        # make_sorting_decision(pieces)


    def recorder_callback(frame):
        """Recording consumer - saves frames to video file"""
        # This would write to video file
        print(f"Recorder saving frame: {frame.shape}")
        # video_writer.write(frame)


    def statistics_callback(frame):
        """Statistics consumer - calculates metrics"""
        # This would compute statistics
        mean_brightness = np.mean(frame)
        print(f"Statistics: brightness = {mean_brightness:.2f}")


    # Register consumers with different priorities
    camera.register_consumer(
        "gui_display",
        gui_display_callback,
        processing_type="async",
        priority=10  # High priority for responsive display
    )

    camera.register_consumer(
        "sorting",
        sorting_algorithm_callback,
        processing_type="sync",  # Synchronous for critical processing
        priority=5
    )

    camera.register_consumer(
        "recorder",
        recorder_callback,
        processing_type="async",
        priority=3
    )

    camera.register_consumer(
        "statistics",
        statistics_callback,
        processing_type="async",
        priority=1  # Low priority
    )

    # Start capture (single stream feeding all consumers)
    camera.start_capture()

    # Let it run for a bit
    time.sleep(5)

    # Print statistics
    stats = camera.get_statistics()
    print("\nCamera Statistics:")
    print(f"  FPS: {stats['camera']['fps_actual']:.2f}")
    print(f"  Total frames: {stats['camera']['total_frames']}")
    print(f"  Dropped frames: {stats['camera']['dropped_frames']}")

    print("\nConsumer Statistics:")
    for name, consumer_stats in stats['consumers'].items():
        print(f"  {name}:")
        print(f"    Frames processed: {consumer_stats['frame_count']}")
        print(f"    Latency: {consumer_stats['latency'] * 1000:.2f}ms")

    # Cleanup
    camera.release()