"""
thread_manager.py - Generic thread management infrastructure

This module provides a reusable thread management system that can be used by ANY module
in the application. It handles thread lifecycle, health monitoring, automatic restart,
and statistics tracking. This is infrastructure code - it knows nothing about Lego pieces
or application-specific logic.

MAIN PURPOSE:
- Provide a consistent way for all modules to manage background threads
- Handle thread failures gracefully with automatic restart capability
- Monitor thread health and performance
- Ensure clean shutdown of all threads

This module is used by:
- piece_queue_manager.py for processing worker threads
- camera_module.py for capture threads
- arduino_servo_module.py for communication threads
- identification_api_handler.py for async API calls
- processing_module.py for background processing
- Any other module that needs managed threads

IMPORT REQUIREMENTS:
- Standard library only (no external dependencies)
- enhanced_config_manager for configuration loading
"""

import threading
import time
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Only external import is from your own modules
from enhanced_config_manager import ModuleConfig, ConfigSchema

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================
# These define the types and structures used throughout the module

class WorkerState(Enum):
    """Possible states for a worker thread"""
    INITIALIZING = "initializing"  # Thread is starting up
    RUNNING = "running"  # Thread is actively working
    PAUSED = "paused"  # Thread is paused but alive
    STOPPING = "stopping"  # Thread is shutting down
    STOPPED = "stopped"  # Thread has exited
    FAILED = "failed"  # Thread crashed
    RESTARTING = "restarting"  # Thread is being restarted


@dataclass
class WorkerConfig:
    """
    Configuration for a managed worker thread.

    This structure holds all the settings needed to manage a worker thread,
    including restart behavior, health checks, and resource limits.
    """
    name: str  # Unique identifier for the worker
    target: Callable  # Function to run in the thread
    args: Tuple = field(default_factory=tuple)  # Arguments for target function
    daemon: bool = True  # Should thread be a daemon?
    restart_on_failure: bool = True  # Auto-restart if thread crashes?
    max_restart_attempts: int = 3  # Maximum restart attempts
    restart_delay: float = 1.0  # Delay between restart attempts
    health_check_interval: float = 5.0  # How often to check thread health
    timeout: float = 30.0  # Maximum time for graceful shutdown
    priority: int = 5  # Thread priority (1-10, higher = more important)


@dataclass
class WorkerStatistics:
    """
    Runtime statistics for a worker thread.

    Tracks performance metrics and health information for monitoring
    and debugging purposes.
    """
    start_time: float = 0.0  # When thread was started
    stop_time: Optional[float] = None  # When thread was stopped
    restart_count: int = 0  # Number of times restarted
    last_restart_time: Optional[float] = None  # Most recent restart
    total_runtime: float = 0.0  # Total time thread has been running
    last_health_check: float = 0.0  # Last health check timestamp
    error_count: int = 0  # Number of errors encountered
    last_error: Optional[str] = None  # Most recent error message
    state: WorkerState = WorkerState.INITIALIZING
    custom_stats: Dict[str, Any] = field(default_factory=dict)  # Module-specific stats


# ============================================================================
# MAIN THREAD MANAGER CLASS
# ============================================================================

class ThreadManager:
    """
    Generic thread manager for the application.

    This class provides a centralized system for managing background threads
    throughout the application. It handles thread lifecycle, monitoring,
    automatic restart on failure, and clean shutdown.

    Key features:
    - Automatic thread restart on failure
    - Health monitoring and statistics
    - Graceful shutdown handling
    - Thread-safe operations
    - Consistent interface for all modules
    """

    def __init__(self, config_manager=None):
        """
        Initialize the thread manager.
        """
        logger.info("Initializing ThreadManager")

        # ====== Configuration Loading ======
        self._load_configuration(config_manager)

        # ====== Thread Storage ======
        self.workers: Dict[str, threading.Thread] = {}
        self.worker_configs: Dict[str, WorkerConfig] = {}
        self.worker_stats: Dict[str, WorkerStatistics] = {}
        self.worker_locks: Dict[str, threading.RLock] = {}

        # ====== Control Flags ======
        self.running = True
        self.shutdown_event = threading.Event()

        # ====== Thread Safety ======
        self._lock = threading.RLock()

        # ============================================================
        # ADD NEW CODE HERE - AFTER EXISTING ATTRIBUTES
        # ============================================================

        # ====== Named Resource Locks (NEW) ======
        # These provide thread safety for shared resources
        self.resource_locks = {
            'api': threading.RLock(),  # For API calls
            'servo': threading.RLock(),  # For servo movements
            'sorting': threading.RLock(),  # For sorting decisions
            'history': threading.RLock(),  # For piece history
        }

        # ====== System Callbacks (NEW) ======
        # For worker lifecycle events
        self.system_callbacks = {
            'worker_started': [],  # When a worker starts
            'worker_stopped': [],  # When a worker stops
            'worker_failed': [],  # When a worker crashes
            'worker_restarted': [],  # When a worker is restarted
            'system_shutdown': [],  # When system is shutting down
        }

        logger.info("ThreadManager initialized with resource locks and callbacks")

        # ============================================================
        # END OF NEW CODE - EXISTING CODE CONTINUES BELOW
        # ============================================================

        # ====== Health Monitoring (EXISTING) ======
        self.health_monitor_thread = None
        if self.enable_health_monitoring:
            self._start_health_monitor()

        # ====== Statistics (EXISTING) ======
        self.global_stats = {
            "total_workers_created": 0,
            "total_restarts": 0,
            "total_failures": 0,
            "manager_start_time": time.time()
        }

        logger.info(f"ThreadManager initialized with health_monitoring={self.enable_health_monitoring}")

    def with_lock(self, resource_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with a resource lock for thread safety.

        This method ensures that only one thread can access a resource at a time,
        preventing race conditions and data corruption.

        Args:
            resource_name: Name of resource to lock ('api', 'servo', 'sorting', 'history')
            func: Function to execute with the lock held
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Whatever the function returns

        Example:
            # Thread-safe API call
            result = thread_manager.with_lock('api', api_client.send_to_api, image_path)

            # Thread-safe servo movement
            success = thread_manager.with_lock('servo', servo.move_to_bin, bin_number)
        """
        # Create lock if it doesn't exist
        if resource_name not in self.resource_locks:
            logger.warning(f"Creating new lock for resource: {resource_name}")
            self.resource_locks[resource_name] = threading.RLock()

        # Get the lock for this resource
        lock = self.resource_locks[resource_name]

        # Execute the function with the lock held
        with lock:
            try:
                # Log for debugging (comment out in production for performance)
                logger.debug(f"Executing {func.__name__} with lock '{resource_name}'")

                # Execute the function
                result = func(*args, **kwargs)

                return result

            except Exception as e:
                logger.error(f"Error executing {func.__name__} with lock '{resource_name}': {e}")
                raise

    def create_lock(self, name: str) -> threading.RLock:
        """
        Create or get a named lock for a resource.

        This allows modules to create their own locks if needed.

        Args:
            name: Name for the lock

        Returns:
            The lock object (RLock for reentrancy)
        """
        if name not in self.resource_locks:
            self.resource_locks[name] = threading.RLock()
            logger.info(f"Created new lock: {name}")

        return self.resource_locks[name]

# ========================================================================
# CALLBACK MANAGEMENT SECTION
# ========================================================================

    def register_system_callback(self, event: str, callback: Callable) -> bool:
        """
        Register a callback for system-level events.

        These are worker lifecycle events, not piece-specific events.

        Args:
            event: Event name ('worker_started', 'worker_stopped', etc.)
            callback: Function to call when event occurs

        Returns:
            True if registered successfully
        """
        if event not in self.system_callbacks:
            logger.warning(f"Unknown system event: {event}")
            return False

        self.system_callbacks[event].append(callback)
        logger.debug(f"Registered system callback for event: {event}")
        return True

    def _trigger_system_callbacks(self, event: str, **kwargs):
        """
        Trigger all callbacks for a system event.

        Args:
            event: Event that occurred
            **kwargs: Event-specific data
        """
        if event not in self.system_callbacks:
            return

        for callback in self.system_callbacks[event]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Error in system callback for {event}: {e}")

    # ========================================================================
    # CONFIGURATION SECTION
    # ========================================================================
    # Methods that handle loading and managing configuration

    def _load_configuration(self, config_manager):
        """
        Load configuration from the config manager or use defaults.

        This method centralizes all configuration loading, ensuring that
        all settings have valid values even if not specified in config.

        Args:
            config_manager: Configuration manager instance or None
        """
        if config_manager:
            # Get complete threading configuration from enhanced_config_manager
            config = config_manager.get_module_config(ModuleConfig.THREADING.value)

            # Extract settings with guaranteed values from schema
            self.max_workers = config.get("max_workers", 10)
            self.default_restart_delay = config.get("worker_restart_delay", 1.0)
            self.health_check_interval = config.get("health_check_interval", 5.0)
            self.shutdown_timeout = config.get("shutdown_timeout", 5.0)
            self.enable_health_monitoring = config.get("enable_health_monitoring", True)
            self.max_restart_attempts = config.get("max_restart_attempts", 3)

            logger.info(f"Loaded configuration from config_manager: max_workers={self.max_workers}")
        else:
            # Use defaults if no config manager provided
            self.max_workers = 10
            self.default_restart_delay = 1.0
            self.health_check_interval = 5.0
            self.shutdown_timeout = 5.0
            self.enable_health_monitoring = True
            self.max_restart_attempts = 3

            logger.info("Using default configuration (no config_manager provided)")

    # ========================================================================
    # WORKER REGISTRATION AND LIFECYCLE SECTION
    # ========================================================================
    # Methods that manage the lifecycle of worker threads

    def register_worker(self, name: str, target: Callable, args: Tuple = (),
                        daemon: bool = True, restart_on_failure: bool = True,
                        max_restart_attempts: Optional[int] = None,
                        priority: int = 5) -> bool:
        """
        Register and start a new worker thread.
        """
        with self._lock:
            # [EXISTING CODE - checking worker limits and existing workers]
            if len(self.workers) >= self.max_workers:
                logger.error(f"Cannot register worker '{name}': max workers ({self.max_workers}) reached")
                return False

            if name in self.workers and self.workers[name].is_alive():
                logger.warning(f"Worker '{name}' already exists and is running")
                return False

            logger.info(f"Registering new worker: {name}")

            # [EXISTING CODE - Create worker configuration]
            config = WorkerConfig(
                name=name,
                target=target,
                args=args,
                daemon=daemon,
                restart_on_failure=restart_on_failure,
                max_restart_attempts=max_restart_attempts or self.max_restart_attempts,
                restart_delay=self.default_restart_delay,
                priority=priority
            )

            # [EXISTING CODE - Initialize statistics]
            stats = WorkerStatistics(
                start_time=time.time(),
                state=WorkerState.INITIALIZING
            )

            # [EXISTING CODE - Store configuration and stats]
            self.worker_configs[name] = config
            self.worker_stats[name] = stats
            self.worker_locks[name] = threading.RLock()

            # [EXISTING CODE - Update global statistics]
            self.global_stats["total_workers_created"] += 1

            # [EXISTING CODE - Define worker wrapper function]
            def worker_wrapper():
                """Wrapper function that handles exceptions and restarts."""
                restart_count = 0

                while self.running:
                    try:
                        # Update state to running
                        with self.worker_locks[name]:
                            self.worker_stats[name].state = WorkerState.RUNNING
                            if restart_count > 0:
                                self.worker_stats[name].last_restart_time = time.time()

                        logger.info(f"Worker '{name}' starting (attempt {restart_count + 1})")

                        # ============================================================
                        # ADD CALLBACK HERE - WHEN WORKER STARTS
                        # ============================================================
                        if restart_count == 0:
                            # First start - trigger worker_started
                            self._trigger_system_callbacks('worker_started', worker_name=name)
                        else:
                            # Restart - trigger worker_restarted
                            self._trigger_system_callbacks('worker_restarted',
                                                           worker_name=name,
                                                           attempt=restart_count)
                        # ============================================================

                        # Run the actual worker function
                        target(*args)

                        # Worker exited normally
                        logger.info(f"Worker '{name}' exited normally")

                        # ============================================================
                        # ADD CALLBACK HERE - WHEN WORKER STOPS NORMALLY
                        # ============================================================
                        self._trigger_system_callbacks('worker_stopped',
                                                       worker_name=name,
                                                       reason='normal_exit')
                        # ============================================================

                        break  # Exit the retry loop

                    except Exception as e:
                        # Worker crashed
                        logger.error(f"Worker '{name}' crashed: {e}", exc_info=True)

                        # Update statistics
                        with self.worker_locks[name]:
                            self.worker_stats[name].error_count += 1
                            self.worker_stats[name].last_error = str(e)
                            self.worker_stats[name].state = WorkerState.FAILED

                        self.global_stats["total_failures"] += 1

                        # ============================================================
                        # ADD CALLBACK HERE - WHEN WORKER FAILS
                        # ============================================================
                        self._trigger_system_callbacks('worker_failed',
                                                       worker_name=name,
                                                       error=str(e),
                                                       restart_count=restart_count)
                        # ============================================================

                        # Check if we should restart
                        if config.restart_on_failure and restart_count < config.max_restart_attempts:
                            restart_count += 1
                            self.worker_stats[name].restart_count = restart_count
                            self.global_stats["total_restarts"] += 1

                            logger.info(f"Restarting worker '{name}' in {config.restart_delay} seconds "
                                        f"(attempt {restart_count}/{config.max_restart_attempts})")

                            # Update state to restarting
                            with self.worker_locks[name]:
                                self.worker_stats[name].state = WorkerState.RESTARTING

                            # Wait before restarting
                            if self.shutdown_event.wait(config.restart_delay):
                                # Shutdown was requested during wait
                                logger.info(f"Shutdown requested, not restarting worker '{name}'")
                                break
                        else:
                            # Max restarts reached or restart not enabled
                            logger.critical(f"Worker '{name}' failed permanently after {restart_count} attempts")

                            # ============================================================
                            # ADD CALLBACK HERE - PERMANENT FAILURE
                            # ============================================================
                            self._trigger_system_callbacks('worker_stopped',
                                                           worker_name=name,
                                                           reason='permanent_failure',
                                                           attempts=restart_count)
                            # ============================================================
                            break

                # Worker is done - update final state
                with self.worker_locks[name]:
                    self.worker_stats[name].state = WorkerState.STOPPED
                    self.worker_stats[name].stop_time = time.time()

            # [EXISTING CODE - Create and start the thread]
            thread = threading.Thread(
                target=worker_wrapper,
                name=f"Worker-{name}",
                daemon=config.daemon
            )

            try:
                thread.start()
                self.workers[name] = thread
                return True
            except Exception as e:
                logger.error(f"Failed to start thread for worker '{name}': {e}")
                return False

    def _start_worker_internal(self, name: str) -> bool:
        """
        Internal method to start a worker thread with the safety wrapper.

        This method wraps the target function in error handling and restart logic.

        Args:
            name: Name of the worker to start

        Returns:
            bool: True if thread was successfully started
        """
        config = self.worker_configs[name]

        def worker_wrapper():
            """
            Wrapper function that runs in the thread and handles failures.

            This wrapper provides:
            - Error catching and logging
            - Automatic restart on failure
            - Statistics tracking
            - Clean shutdown handling
            """
            restart_count = 0

            while self.running and restart_count <= config.max_restart_attempts:
                try:
                    # Update state to running
                    with self.worker_locks[name]:
                        self.worker_stats[name].state = WorkerState.RUNNING
                        if restart_count > 0:
                            self.worker_stats[name].last_restart_time = time.time()

                    logger.info(f"Worker '{name}' starting (attempt {restart_count + 1})")

                    # Call the actual target function
                    # The target function should check self.shutdown_event periodically
                    config.target(*config.args)

                    # If we get here, the function exited normally
                    logger.info(f"Worker '{name}' exited normally")
                    break

                except Exception as e:
                    # Worker crashed - log the error
                    logger.error(f"Worker '{name}' crashed with error: {e}", exc_info=True)

                    # Update statistics
                    with self.worker_locks[name]:
                        self.worker_stats[name].error_count += 1
                        self.worker_stats[name].last_error = str(e)
                        self.worker_stats[name].state = WorkerState.FAILED

                    self.global_stats["total_failures"] += 1

                    # Check if we should restart
                    if config.restart_on_failure and restart_count < config.max_restart_attempts:
                        restart_count += 1
                        self.worker_stats[name].restart_count = restart_count
                        self.global_stats["total_restarts"] += 1

                        logger.info(f"Restarting worker '{name}' in {config.restart_delay} seconds "
                                    f"(attempt {restart_count}/{config.max_restart_attempts})")

                        # Update state to restarting
                        with self.worker_locks[name]:
                            self.worker_stats[name].state = WorkerState.RESTARTING

                        # Wait before restarting (unless we're shutting down)
                        if self.shutdown_event.wait(config.restart_delay):
                            # Shutdown was requested during wait
                            logger.info(f"Shutdown requested, not restarting worker '{name}'")
                            break
                    else:
                        # Max restarts reached or restart not enabled
                        logger.critical(f"Worker '{name}' failed permanently after {restart_count} attempts")
                        break

            # Worker is done - update final state
            with self.worker_locks[name]:
                self.worker_stats[name].state = WorkerState.STOPPED
                self.worker_stats[name].stop_time = time.time()

        # Create and start the thread with the wrapper
        thread = threading.Thread(
            target=worker_wrapper,
            name=f"Worker-{name}",
            daemon=config.daemon
        )

        try:
            thread.start()
            self.workers[name] = thread
            return True
        except Exception as e:
            logger.error(f"Failed to start thread for worker '{name}': {e}")
            return False

    def unregister_worker(self, name: str, timeout: Optional[float] = None) -> bool:
        """
        Stop and remove a worker thread.

        This method gracefully stops a worker thread and removes it from management.

        Args:
            name: Name of the worker to stop
            timeout: Maximum time to wait for thread to stop (uses default if None)

        Returns:
            bool: True if worker was successfully stopped and removed
        """
        with self._lock:
            if name not in self.workers:
                logger.warning(f"Cannot unregister worker '{name}': not found")
                return False

            logger.info(f"Unregistering worker: {name}")

            # Update state
            with self.worker_locks[name]:
                self.worker_stats[name].state = WorkerState.STOPPING

            # Get the thread
            thread = self.workers[name]

            # Use provided timeout or default
            timeout = timeout or self.shutdown_timeout

            # Wait for thread to exit
            thread.join(timeout=timeout)

            if thread.is_alive():
                logger.warning(f"Worker '{name}' did not stop within {timeout} seconds")
                return False

            # Clean up
            del self.workers[name]
            del self.worker_configs[name]

            # Keep stats for debugging but mark as stopped
            with self.worker_locks[name]:
                self.worker_stats[name].state = WorkerState.STOPPED
                self.worker_stats[name].stop_time = time.time()

            logger.info(f"Successfully unregistered worker: {name}")
            return True

    def pause_worker(self, name: str) -> bool:
        """
        Pause a worker thread (if the worker supports pausing).

        Note: The worker's target function must check for pause state.

        Args:
            name: Name of the worker to pause

        Returns:
            bool: True if pause was requested successfully
        """
        with self._lock:
            if name not in self.workers:
                logger.warning(f"Cannot pause worker '{name}': not found")
                return False

            with self.worker_locks[name]:
                self.worker_stats[name].state = WorkerState.PAUSED

            logger.info(f"Paused worker: {name}")
            return True

    def resume_worker(self, name: str) -> bool:
        """
        Resume a paused worker thread.

        Args:
            name: Name of the worker to resume

        Returns:
            bool: True if resume was requested successfully
        """
        with self._lock:
            if name not in self.workers:
                logger.warning(f"Cannot resume worker '{name}': not found")
                return False

            with self.worker_locks[name]:
                if self.worker_stats[name].state != WorkerState.PAUSED:
                    logger.warning(f"Worker '{name}' is not paused")
                    return False

                self.worker_stats[name].state = WorkerState.RUNNING

            logger.info(f"Resumed worker: {name}")
            return True

    # ========================================================================
    # HEALTH MONITORING SECTION
    # ========================================================================
    # Methods that monitor thread health and handle failures

    def _start_health_monitor(self):
        """
        Start the health monitoring thread.

        This thread periodically checks the health of all worker threads
        and can trigger restarts if threads have died unexpectedly.
        """

        def health_monitor_loop():
            """Main loop for health monitoring thread."""
            logger.info("Health monitor started")

            while self.running:
                # Wait for the check interval or shutdown signal
                if self.shutdown_event.wait(self.health_check_interval):
                    # Shutdown requested
                    break

                # Check health of all workers
                self._check_all_workers_health()

            logger.info("Health monitor stopped")

        self.health_monitor_thread = threading.Thread(
            target=health_monitor_loop,
            name="HealthMonitor",
            daemon=True
        )
        self.health_monitor_thread.start()

    def _check_all_workers_health(self):
        """
        Check the health of all registered workers.

        This method is called periodically by the health monitor thread.
        It detects dead threads and can trigger automatic restarts.
        """
        with self._lock:
            current_time = time.time()

            for name in list(self.workers.keys()):
                thread = self.workers[name]
                config = self.worker_configs[name]
                stats = self.worker_stats[name]

                # Update last health check time
                with self.worker_locks[name]:
                    stats.last_health_check = current_time

                # Check if thread is alive
                if not thread.is_alive():
                    # Thread has died
                    if stats.state == WorkerState.RUNNING:
                        # Thread died unexpectedly
                        logger.warning(f"Worker '{name}' found dead during health check")

                        with self.worker_locks[name]:
                            stats.state = WorkerState.FAILED
                            stats.error_count += 1

                        # Check if we should restart
                        if config.restart_on_failure and stats.restart_count < config.max_restart_attempts:
                            logger.info(f"Attempting to restart worker '{name}'")
                            self._start_worker_internal(name)
                        else:
                            logger.error(f"Worker '{name}' will not be restarted "
                                         f"(restart_count={stats.restart_count}, max={config.max_restart_attempts})")

    def get_worker_health(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get health status of a specific worker.

        Args:
            name: Name of the worker to check

        Returns:
            Dictionary with health information or None if worker not found

        Example return value:
            {
                "alive": True,
                "state": "running",
                "uptime": 123.45,
                "restart_count": 0,
                "error_count": 0,
                "last_error": None
            }
        """
        with self._lock:
            if name not in self.workers:
                return None

            thread = self.workers[name]
            stats = self.worker_stats[name]

            with self.worker_locks[name]:
                uptime = time.time() - stats.start_time
                if stats.stop_time:
                    uptime = stats.stop_time - stats.start_time

                return {
                    "alive": thread.is_alive(),
                    "state": stats.state.value,
                    "uptime": uptime,
                    "restart_count": stats.restart_count,
                    "error_count": stats.error_count,
                    "last_error": stats.last_error,
                    "last_health_check": stats.last_health_check
                }

    # ========================================================================
    # STATISTICS AND MONITORING SECTION
    # ========================================================================
    # Methods for tracking performance and gathering statistics

    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all workers and the manager itself.

        Returns:
            Dictionary containing detailed statistics

        Example return value:
            {
                "manager": {
                    "uptime": 3600.0,
                    "total_workers_created": 5,
                    "total_restarts": 2,
                    "total_failures": 3
                },
                "workers": {
                    "api_processor": { ... },
                    "camera_capture": { ... }
                }
            }
        """
        with self._lock:
            # Calculate manager uptime
            manager_uptime = time.time() - self.global_stats["manager_start_time"]

            # Gather worker statistics
            worker_stats = {}
            for name in self.workers:
                worker_stats[name] = self.get_worker_health(name)

            return {
                "manager": {
                    "uptime": manager_uptime,
                    "total_workers_created": self.global_stats["total_workers_created"],
                    "total_restarts": self.global_stats["total_restarts"],
                    "total_failures": self.global_stats["total_failures"],
                    "active_workers": len(self.workers),
                    "max_workers": self.max_workers
                },
                "workers": worker_stats
            }

    def update_worker_custom_stats(self, name: str, stats: Dict[str, Any]):
        """
        Update custom statistics for a specific worker.

        This allows workers to report application-specific metrics.

        Args:
            name: Name of the worker
            stats: Dictionary of custom statistics to update

        Example:
            thread_manager.update_worker_custom_stats("api_processor", {
                "requests_processed": 1234,
                "average_response_time": 0.5,
                "cache_hit_rate": 0.85
            })
        """
        with self._lock:
            if name in self.worker_stats:
                with self.worker_locks[name]:
                    self.worker_stats[name].custom_stats.update(stats)

    def get_worker_custom_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get custom statistics for a specific worker.

        Args:
            name: Name of the worker

        Returns:
            Dictionary of custom statistics or None if worker not found
        """
        with self._lock:
            if name in self.worker_stats:
                with self.worker_locks[name]:
                    return self.worker_stats[name].custom_stats.copy()
            return None

    # ========================================================================
    # SHUTDOWN AND CLEANUP SECTION
    # ========================================================================
    # Methods for graceful shutdown and resource cleanup

    def shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Gracefully shutdown all worker threads and the thread manager.

        This method:
        1. Signals all threads to stop
        2. Waits for threads to exit gracefully
        3. Cleans up resources

        Args:
            timeout: Maximum time to wait for all threads to stop

        Returns:
            bool: True if all threads stopped cleanly, False if some had to be abandoned
        """
        logger.info("Starting ThreadManager shutdown")

        # Use provided timeout or default
        timeout = timeout or self.shutdown_timeout

        # Signal shutdown to all threads
        self.running = False
        self.shutdown_event.set()

        # Stop health monitor first
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=2.0)

        # Calculate timeout per worker based on priority
        workers_by_priority = sorted(
            self.worker_configs.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )

        all_stopped = True
        remaining_timeout = timeout
        start_time = time.time()

        # Stop workers in priority order
        for name, config in workers_by_priority:
            if name not in self.workers:
                continue

            thread = self.workers[name]
            if not thread.is_alive():
                continue

            # Calculate timeout for this worker
            elapsed = time.time() - start_time
            remaining_timeout = max(0.1, timeout - elapsed)
            worker_timeout = min(remaining_timeout, config.timeout)

            logger.info(f"Stopping worker '{name}' (priority={config.priority}, timeout={worker_timeout:.1f}s)")

            # Update state
            with self.worker_locks[name]:
                self.worker_stats[name].state = WorkerState.STOPPING

            # Wait for thread to stop
            thread.join(timeout=worker_timeout)

            if thread.is_alive():
                logger.warning(f"Worker '{name}' did not stop within timeout")
                all_stopped = False
            else:
                logger.info(f"Worker '{name}' stopped successfully")
                with self.worker_locks[name]:
                    self.worker_stats[name].state = WorkerState.STOPPED
                    self.worker_stats[name].stop_time = time.time()

        # Log final statistics
        final_stats = self.get_all_statistics()
        logger.info(f"ThreadManager shutdown complete. Final stats: {final_stats['manager']}")

        return all_stopped

    def get_shutdown_event(self) -> threading.Event:
        """
        Get the shutdown event that workers can monitor.

        Workers should periodically check this event and exit cleanly when set.

        Returns:
            The shutdown event

        Example usage in a worker function:
            def my_worker(shutdown_event):
                while not shutdown_event.is_set():
                    # Do work...
                    if shutdown_event.wait(timeout=0.1):
                        break  # Shutdown requested
        """
        return self.shutdown_event

    def is_running(self) -> bool:
        """
        Check if the thread manager is running.

        Returns:
            bool: True if running, False if shutdown has been initiated
        """
        return self.running


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_thread_manager(config_manager=None) -> ThreadManager:
    """
    Create a thread manager instance.

    This is the recommended way to create a ThreadManager instance.

    Args:
        config_manager: Optional configuration manager

    Returns:
        ThreadManager instance

    Example:
        config_manager = create_config_manager()
        thread_manager = create_thread_manager(config_manager)

        # Register workers
        thread_manager.register_worker(
            name="my_worker",
            target=my_worker_function,
            args=(arg1, arg2),
            restart_on_failure=True
        )
    """
    return ThreadManager(config_manager)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating how to use the ThreadManager.

    This shows how other modules would integrate with the thread manager.
    """


    # Example worker function
    def example_worker(worker_id: int, shutdown_event: threading.Event):
        """Example worker that does some processing."""
        logger.info(f"Worker {worker_id} started")

        counter = 0
        while not shutdown_event.is_set():
            # Simulate work
            time.sleep(1)
            counter += 1
            logger.info(f"Worker {worker_id}: Processed item {counter}")

            # Check for shutdown every second
            if shutdown_event.wait(timeout=0):
                break

        logger.info(f"Worker {worker_id} stopping")


    # Set up logging for the example
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create thread manager
    manager = create_thread_manager()

    # Register some workers
    for i in range(3):
        manager.register_worker(
            name=f"worker_{i}",
            target=example_worker,
            args=(i, manager.get_shutdown_event()),
            restart_on_failure=True,
            priority=5 + i
        )

    # Let workers run for a bit
    logger.info("Workers running... Press Ctrl+C to stop")

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    # Get statistics before shutdown
    stats = manager.get_all_statistics()
    logger.info(f"Statistics: {stats}")

    # Shutdown
    manager.shutdown(timeout=5.0)
    logger.info("Example complete")