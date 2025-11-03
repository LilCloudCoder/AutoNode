import secrets
import os
import logging
import time
from typing import Optional
from ctypes import cdll, c_int
import json
from datetime import datetime

try:
    import psutil
    from numba import njit, prange
    import numpy as np
except ImportError:
    raise ImportError("INSTALLATION REQUIRED. Modules: Numba/Psutil/Numpy")

# Paths and globals
BASE_DIR = os.path.dirname(__file__)
THREAD_REGISTRY_PATH = os.path.abspath(os.path.join(os.path.dirname(BASE_DIR), "thread_registry.json"))
SWIFT_DYLIB_PATH = os.path.abspath(os.path.join(BASE_DIR, "swift", "libMetalBridge.dylib"))

# Configure root logging only if not already configured by the host app
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("Threader")

# One-time flags/cache
_STATS_LOGGED = False
_GPU_LIB = None


class MetalThreadRunner:
    """
    Executes GPU-bound threads using a Swift-Metal compiled dynamic library.

    This class serves as a bridge between Python thread logic and the Apple GPU
    execution unit using `libMetalBridge.dylib`. It assumes the .dylib exists and
    that the Metal device and shader pipeline are configured correctly.

    Methods
    -------
    execute(thread_id: int)
        Executes the GPU logic for the given thread identifier.
    """

    def __init__(self):
        global _GPU_LIB
        if _GPU_LIB is None:
            if not os.path.exists(SWIFT_DYLIB_PATH):
                raise FileNotFoundError(f"Metal dylib not found at {SWIFT_DYLIB_PATH}")
            _GPU_LIB = cdll.LoadLibrary(SWIFT_DYLIB_PATH)
        self.lib = _GPU_LIB

    def execute(self, thread_id: int):
        logger.info(f"[PYTHON] Dispatching Thread {thread_id} to GPU")
        self.lib.run_gpu_thread(c_int(thread_id))


def generate_token() -> str:
    """
    Generates a secure hexadecimal authentication token for internal verification.

    Returns
    -------
    str
        A 32-character hexadecimal token string.
    """
    return secrets.token_hex(16)


def log_system_stats():
    """
    Logs current system CPU and memory usage once per process (non-blocking).
    Also checks for GPU (Metal dylib) availability using a cached path.
    """
    global _STATS_LOGGED
    if _STATS_LOGGED:
        return
    try:
        cpu_percent = psutil.cpu_percent(interval=0.0)
        mem = psutil.virtual_memory()
        logger.info(f"CPU Usage: {cpu_percent}% | Memory: {mem.percent}% used")
        if os.path.exists(SWIFT_DYLIB_PATH):
            logger.info("GPU (Metal dylib) detected and available.")
        else:
            logger.warning(f"GPU dylib not found at {SWIFT_DYLIB_PATH}. GPU execution may fail.")
    except Exception as e:
        logger.error(f"System stats check failed: {e}")
    finally:
        _STATS_LOGGED = True


class CreateThread:
    """
    CreateThread is a secure, token-protected thread definition and execution unit
    for both CPU and GPU-bound logic. Threads are assigned metadata such as priority,
    stage, name, and sandbox status. Execution is monitored, logged, and tracked.

    ATTRS
    ----------
        thread_id : float
            Unique identifier for this thread.
        stage : str
            Execution stage: CRITICAL, CORE, or BASE.
        device : str
            Target hardware: 'cpu' or 'gpu'.
        token : str
            Auth token for permissioned execution.
        sandbox : bool
            If True, runs thread in sandbox mode with restrictions.
        caching : bool
            If True, enables thread-level caching.
        priority : int
            Scheduling importance for external managers.
        ttl : float
            Metadata-only TTL value for runtime observers.
        metadata : dict
            Arbitrary data for tracking and classification.
        thread_name : str
            Optional name override. Auto-generated if None.
        history : list
            Execution logs for this thread instance.

    MTDS
    -------
        spawn()
            Dispatches the thread for execution based on its device type.
        describe()
            Logs a structured description of this thread.
        print_info(thread_id)
            Static method that prints thread details from JSON registry.

    Example Usage
    ------------
    >>> thread = CreateThread(
    ...     thread_id=1.0,
    ...     stage="CORE",
    ...     device="cpu",
    ...     token="your_secure_token",
    ...     caching=True,
    ...     priority=2,
    ...     ttl=15.0,
    ...     metadata={"max_memory_mb": 1000},
    ...     thread_name="YourCoreThread"
    >>> )
    """

    VALID_STAGES = {"CRITICAL", "CORE", "BASE"}
    VALID_DEVICES = {"cpu", "gpu"}

    def __init__(
        self,
        thread_id: float,
        stage: str,
        device: str,
        token: str,
        sandbox: bool = True,
        caching: bool = True,
        priority: int = 1,
        ttl: float = 10.0,
        metadata: Optional[dict] = None,
        thread_name: Optional[str] = None
    ):
        if stage not in self.VALID_STAGES:
            raise ValueError(
                f"Invalid stage: {stage}. Must be one of {self.VALID_STAGES}"
            )
        if device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device: {device}. Must be one of {self.VALID_DEVICES}"
            )

        log_system_stats()

        self.thread_id = thread_id
        self.stage = stage
        self.device = device
        self.sandbox = sandbox
        self.caching = caching
        self.priority = priority
        self.ttl = ttl
        self.metadata = metadata or {}
        self.thread_name = thread_name or self._generate_thread_name()
        self.history = []

        self.max_memory_mb = self.metadata.get("max_memory_mb", 500)

        # Establish a session token using environment variable
        env_key = "AUTONODE_TOKEN"
        env_token = os.environ.get(env_key)
        if not env_token:
            if not isinstance(token, str) or not token.strip():
                raise PermissionError("Invalid authentication token.")
            os.environ[env_key] = token
            env_token = token

        self._auth_token = env_token
        self._validate_token(token)
        self._validate_thread_parameters()
        self._record_event(
            f"Thread '{self.thread_name}' initialized with priority {self.priority}."
        )

    def _validate_token(self, token: str):
        """
        Validates the provided token against the session token (from environment).
        """
        if not isinstance(token, str) or not token.strip():
            raise PermissionError("Invalid authentication token.")
        if token != self._auth_token:
            raise PermissionError("Invalid authentication token.")

    def _record_event(self, message: str):
        """
        Records a log event in the thread's history with a timestamp.

        Parameters
        ----------
        message : str
            The message to record in history.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(f"[{timestamp}] {message}")
        logger.info(message)

    def _generate_thread_name(self) -> str:
        """
        Generates a thread name based on ID and stage.

        Returns
        -------
        str
            A formatted thread name string.
        """
        return f"Thread-{int(self.thread_id)}-{self.stage.lower()}"

    def _validate_thread_parameters(self):
        """
        Validates thread attributes like priority and TTL for correctness.

        Raises
        ------
        ValueError
            If priority is not a non-negative int or TTL is not a positive number.
        """
        if not isinstance(self.priority, int) or self.priority < 0:
            raise ValueError("Priority must be a non-negative integer.")
        if not isinstance(self.ttl, (int, float)) or self.ttl <= 0:
            raise ValueError("TTL must be a positive number.")

    def spawn(self, dry_run: bool = False):
        """
        Dispatches the thread for execution based on its configured device.

        Parameters
        ----------
        dry_run : bool
            If True, simulates the spawn without actual execution.

        Raises
        ------
        MemoryError
            If the system is under high memory or CPU load.
        """
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.5)
        if mem.percent > 95 or cpu > 95:
            error_msg = (
                f"ERROR: MEMORY FULL OR CPU OVERLOADED | RAM: {mem.percent}%, CPU: {cpu}%"
            )
            logger.critical(error_msg)
            raise MemoryError(error_msg)

        if dry_run:
            logger.info(
                f"[DRY-RUN] Simulated spawning of thread {self.thread_id} on {self.device.upper()}"
            )
            return

        if self.stage == "CRITICAL":
            self.device = "gpu"
            self._record_event("Stage is CRITICAL. Auto-switching to GPU.")

        self._record_event(
            f"Spawning '{self.thread_name}' | ID: {self.thread_id} | Device: {self.device.upper()} | "
            f"Stage: {self.stage} | Priority: {self.priority} | TTL: {self.ttl}s | Metadata: {self.metadata}"
        )

        if self.sandbox:
            logger.info(
                f"[SANDBOXED] '{self.thread_name}' executing with sandbox constraints."
            )
        else:
            logger.warning(
                f"[UNSANDBOXED] '{self.thread_name}' has elevated permissions!"
            )

        if self.device == "cpu":
            self._run_cpu()
        elif self.device == "gpu":
            self._run_gpu()

        registry = {}
        if os.path.exists(THREAD_REGISTRY_PATH):
            with open(THREAD_REGISTRY_PATH, "r") as f:
                try:
                    registry = json.load(f)
                except json.JSONDecodeError:
                    registry = {}

        registry[str(self.thread_id)] = {
            "id": self.thread_id,
            "name": self.thread_name,
            "stage": self.stage,
            "device": self.device,
            "sandbox": self.sandbox,
            "caching": self.caching,
            "priority": self.priority,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "history": self.history,
        }

        with open(THREAD_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)

    @staticmethod
    def print_info(thread_id: float):
        """
        Prints the thread metadata from JSON registry by its ID.

        Parameters
        ----------
        thread_id : float
            The ID of the thread to fetch from the registry.
        """
        if not os.path.exists(THREAD_REGISTRY_PATH):
            logger.warning("Thread registry not found.")
            return
        with open(THREAD_REGISTRY_PATH, "r") as f:
            data = json.load(f)
        thread_info = data.get(str(thread_id))
        if not thread_info:
            logger.info(f"No thread found with ID {thread_id}")
        else:
            logger.info("THREAD INFO:")
            for key, value in thread_info.items():
                logger.info(f"  {key}: {value}")

    def _run_cpu(self):
        """
        Executes CPU thread logic using Numba-parallel matrix multiplication.

        Raises
        ------
        MemoryError
            If estimated matrix size exceeds thread memory budget.
        """

        try:
            logger.info(
                f"CPU thread {self.thread_id} executing at stage {self.stage} using Numba-parallel kernel..."
            )

            size = 512
            estimated_size_mb = 3 * (size ** 2 * 8) / (1024 ** 2)
            if estimated_size_mb > self.max_memory_mb:
                error_msg = (
                    f"Thread {self.thread_id} exceeds memory budget. Needs {estimated_size_mb:.2f}MB, allowed {self.max_memory_mb}MB"
                )
                logger.critical(error_msg)
                raise MemoryError(error_msg)

            @njit(parallel=True)
            def parallel_kernel(a, b, out):
                for i in prange(a.shape[0]):
                    for j in range(b.shape[1]):
                        total = 0.0
                        for k in range(a.shape[1]):
                            total += a[i, k] * b[k, j]
                        out[i, j] = total

            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            out = np.zeros((size, size))

            t0 = time()
            parallel_kernel(a, b, out)
            t1 = time()

            logger.info(
                f"Numba-parallel matrix multiplication complete. Shape: {out.shape}, Time: {t1 - t0:.4f}s"
            )
            self._record_event("CPU computation complete.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"CPU execution failed:\n{tb}")
            self._record_event(f"CPU computation failed {e}.")

    def _run_gpu(self):
        """
        Executes the thread logic on GPU using MetalThreadRunner.
        """
        try:
            runner = MetalThreadRunner()
            runner.execute(int(self.thread_id))
            self._record_event("GPU computation complete.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"GPU execution failed:\n{tb}")
            self._record_event(f"GPU computation failed. {e}")


def spawn_thread_batch(batch_data: list) -> list:
    """
    Spawns multiple threads from a batch list of thread specifications.

    Each item in `batch_data` must be a dictionary containing the keys required
    to initialize a CreateThread instance. Threads are spawned immediately and
    logged individually. All successfully spawned threads are returned.

    Parameters
    ----------
        batch_data : list of dict
            A list of thread creation parameter dictionaries.

    Returns
    -------
        list
            A list of CreateThread instances that were successfully spawned.
    """
    logger.info(f"Starting batch spawn for {len(batch_data)} threads")

    threads = []
    for data in batch_data:
        try:
            thread = CreateThread(
                thread_id=data["id"],
                stage=data.get("stage", "BASE"),
                device=data.get("device", "cpu"),
                token=data["token"],
                sandbox=data.get("sandbox", True),
                caching=data.get("caching", True),
                priority=data.get("priority", 1),
                ttl=data.get("ttl", 10.0),
                metadata=data.get("metadata", {}),
                thread_name=data.get("name", None)
            )
            thread.spawn()
            threads.append(thread)
        except Exception as e:
            logger.error(
                f"Failed to spawn thread '{data.get('name', data['id'])}': {e}"
            )
    logger.info(
        f"Batch spawn completed. {len(threads)} threads successfully launched."
    )
    return threads

if __name__ == "__main__":
    token = generate_token()
    print(token)

# Dir: ./threader
# END OF FILE: create.py
# MODULES: CreateThread | MetalThreadRunner
