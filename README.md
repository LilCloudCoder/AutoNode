# AutoNode

A lightweight Python framework for creating, managing, and executing token-protected "threads" (lightweight tasks) on CPU or Apple GPU (Metal). It includes:

- A `CreateThread` runtime to define and run threads with metadata, priority, sandboxing, and caching.
- A JSON-backed thread registry with utilities to list, query, branch, fork, merge, reparent, rename, and mutate threads.
- Optional GPU execution path via a Swift/Metal dynamic library (`libMetalBridge.dylib`).
- Simple CLI to delete threads with archival.

> Current date: 2025-11-03

---

## Features

- Secure token validation per thread instance
- CPU execution using Numba for parallel kernels
- Optional GPU dispatch through Swift/Metal bridge
- Persistent JSON registry at `./thread_registry.json`
- Rich history trail per thread (timestamps)
- Batch spawning utility
- Advanced registry operations: branch, merge (with conflict strategies), fork, reparent, rename, mutate

---

## Project Layout

```
AutoNode/
├─ core/
│  ├─ create.py              # Thread creation, execution (CPU/GPU), batch utilities
│  ├─ operations.py          # Registry utilities: list/get/branch/merge/fork/etc.
│  ├─ delete.py              # CLI to delete and archive threads
│  ├─ router.py              # (placeholder)
│  └─ swift/
│     ├─ GPU.swift           # Swift GPU logic
│     ├─ MetalBridge.swift   # Swift/C bridge to expose GPU entrypoints
│     └─ libMetalBridge.dylib# Compiled dynamic library used by Python via ctypes
├─ main.py                   # Minimal usage example
├─ pyrightconfig.json        # Type checking configuration
├─ requirements.txt          # Python deps (psutil, numba, numpy)
└─ .gitignore
```

---

## Requirements

- Python 3.9+
- macOS with Apple Metal (for GPU path)
- Python packages (see `requirements.txt`):
  - `psutil`
  - `numba`
  - `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## GPU Support (Metal)

- The Python side loads a dynamic library via `ctypes` to dispatch a GPU thread.
- Expected dylib location for this repository: `core/swift/libMetalBridge.dylib`.
- The class `MetalThreadRunner` in `core/create.py` attempts to load a dylib by relative path. Ensure the path used in code matches the actual file path above. If necessary, update the loader path or place a copy/symlink where expected.
- Swift sources are in `core/swift/`. You’ll need to compile them into `libMetalBridge.dylib` appropriate for your system if you modify them.

Example loader expectation in code:

```python
# In core/create.py
self.lib = cdll.LoadLibrary(os.path.abspath("./swift/libMetalBridge.dylib"))
```

If your dylib lives at `core/swift/libMetalBridge.dylib`, update the path accordingly or run Python from the `core/` directory.

---

## Quick Start

1) Generate a token and create a thread (CPU):

```python
# main.py
from core.create import CreateThread, generate_token

token = generate_token()

thread = CreateThread(
    thread_id=1.0,
    stage="CORE",      # one of: CRITICAL | CORE | BASE
    device="cpu",      # or "gpu" (requires Metal dylib)
    token=token,
    caching=True,
    priority=2,
    ttl=15.0,
    metadata={"max_memory_mb": 1000},
    thread_name="YourCoreThread",
)

# To actually run:
thread.spawn(dry_run=False)
```

2) Inspect the thread registry:

```python
from core.operations import list_threads, get_thread

all_threads = list_threads()
print(all_threads)

t = get_thread(1.0)
print(t)
```

3) Delete a thread via CLI (with archival):

```bash
python -m core.delete 1.0 2.0 3.0
```

- Registry file: `./thread_registry.json`
- Deleted backup: `./deleted_threads.json`

---

## `CreateThread` Overview (core/create.py)

Constructor:

```python
CreateThread(
    thread_id: float,
    stage: str,           # "CRITICAL" | "CORE" | "BASE"
    device: str,          # "cpu" | "gpu"
    token: str,           # auth token
    sandbox: bool = True,
    caching: bool = True,
    priority: int = 1,
    ttl: float = 10.0,
    metadata: Optional[dict] = None,
    thread_name: Optional[str] = None,
)
```

Key behaviors:

- Validates token format
- Auto-generates name if not provided (e.g., `Thread-<id>-<stage-lower>`) 
- Records history events with timestamps
- `spawn(dry_run: bool = False)`
  - Validates parameters
  - Records start/finish events, execution duration
  - Runs CPU kernel (Numba-parallelized example) or dispatches GPU via `MetalThreadRunner`
  - Persists/merges entry in `thread_registry.json`
- `print_info(thread_id: float)` to print registry info for a specific thread
- `spawn_thread_batch(batch_data: list)` helper to launch a batch of thread definitions

---

## Registry Operations (core/operations.py)

Utilities operate on `thread_registry.json` and preserve a `history` field with timestamped entries.

- `list_threads() -> List[Dict[str, Any]]`
- `get_thread(thread_id: float) -> Optional[Dict[str, Any]]`
- `branch_thread(source_id: float, new_id: float, overrides: Optional[Dict[str, Any]] = None)`
  - Copies base fields/metadata, applies overrides, updates lineage and history
- `merge_threads(left_id: float, right_id: float, new_id: Optional[float] = None, strategy: str = "prefer-left", conflict_keys: Optional[List[str]] = None)`
  - Merge strategies: `prefer-left` | `prefer-right` | `manual`
  - Optional `conflict_keys` to force manual resolution on specific keys
- `fork_thread(source_id: float, new_ids: List[float], per_thread_overrides: Optional[List[Optional[Dict[str, Any]]]] = None)`
- `reparent_thread(child_id: float, new_parent_id: float)`
- `rename_thread(thread_id: float, new_name: str)`
- `mutate_thread(thread_id: float, updates: Dict[str, Any], safe_fields: Optional[List[str]] = None)`
- `summarize_registry()` -> Dict with counts and lineage summary

---

## CPU vs GPU Execution

- CPU path (`device="cpu"`):
  - Demonstrates a Numba-accelerated parallel kernel combining two arrays.
  - Logs runtime stats and writes registry entry.

- GPU path (`device="gpu"`):
  - Requires Metal; uses `MetalThreadRunner` to `LoadLibrary` the Swift/Metal dylib and call `run_gpu_thread`.
  - Ensure the compiled dylib exists at the expected path.

---

## Development Notes

- Logging is configured with a consistent format across modules.
- `pyrightconfig.json` is included for type checking; run Pyright or your IDE for static analysis.
- `core/router.py` is currently a placeholder for potential future routing/orchestration logic.
- Be mindful of the relative path used when loading `libMetalBridge.dylib`; align it with your run working directory or switch to a robust path join based on `__file__`.

---

## Troubleshooting

- ImportError: "INSTALLATION REQUIRED. Modules: Numba/Psutil/Numpy"
  - Install dependencies with `pip install -r requirements.txt`.
- GPU dylib not found
  - Confirm `core/swift/libMetalBridge.dylib` exists; adjust loader path or working directory.
- Corrupted `thread_registry.json`
  - `operations._load_registry` will recover with an empty registry and log a warning.

---

## License

MIT (or project default). Update this section if a different license applies.
