import os
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from copy import deepcopy

THREAD_REGISTRY_PATH = os.path.abspath("./thread_registry.json")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger("ThreadOperations")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(THREAD_REGISTRY_PATH):
        return {}
    try:
        with open(THREAD_REGISTRY_PATH, "r") as f:
            return json.load(f) or {}
    except json.JSONDecodeError:
        logger.warning("Thread registry is corrupted; starting with an empty registry.")
        return {}


def _save_registry(registry: Dict[str, Any]) -> None:
    with open(THREAD_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _append_history(entry: Dict[str, Any], message: str) -> None:
    entry.setdefault("history", [])
    entry["history"].append(f"[{_now()}] {message}")


def list_threads() -> List[Dict[str, Any]]:
    """
    Returns a list of all threads from the registry.
    """
    reg = _load_registry()
    return list(reg.values())


def get_thread(thread_id: float) -> Optional[Dict[str, Any]]:
    """
    Returns a single thread by id or None if not found.
    """
    reg = _load_registry()
    return reg.get(str(thread_id))


def branch_thread(
    source_id: float,
    new_id: float,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a new thread entry by branching from an existing thread.
    - Copies metadata and most fields.
    - Applies optional overrides.
    - Adds history note indicating the branch.

    Returns the created thread entry.
    """
    reg = _load_registry()
    key = str(source_id)
    if key not in reg:
        raise KeyError(f"Source thread {source_id} not found.")
    if str(new_id) in reg:
        raise ValueError(f"Target id {new_id} already exists.")

    base = deepcopy(reg[key])

    # Minimal normalized copy
    branched = {
        "id": new_id,
        "name": overrides.get("name") if overrides else None,
        "stage": base.get("stage"),
        "device": base.get("device"),
        "sandbox": base.get("sandbox", True),
        "caching": base.get("caching", True),
        "priority": base.get("priority", 1),
        "ttl": base.get("ttl", 10.0),
        "metadata": deepcopy(base.get("metadata", {})),
        "history": deepcopy(base.get("history", [])),
        "parent": base.get("parent", base.get("id")),
        "lineage": deepcopy(base.get("lineage", [])),
    }

    # Default name if not provided
    if not branched["name"]:
        stage = str(branched.get("stage", "BASE")).lower()
        branched["name"] = f"Thread-{int(new_id)}-{stage}"

    # Update lineage
    parent_id = base.get("id")
    if parent_id is not None:
        branched["lineage"].append(parent_id)

    # Apply overrides safely
    if overrides:
        for k, v in overrides.items():
            if k in ("id",):  # do not allow id override here
                continue
            if k == "metadata":
                # Deep merge metadata
                md = branched.get("metadata", {})
                md = {**md, **(v or {})}
                branched["metadata"] = md
            else:
                branched[k] = v

    _append_history(branched, f"Branched from {source_id} into {new_id}.")

    reg[str(new_id)] = branched
    _save_registry(reg)
    logger.info(f"Branched thread {source_id} -> {new_id}")
    return branched


def merge_threads(
    left_id: float,
    right_id: float,
    new_id: Optional[float] = None,
    strategy: str = "prefer-left",
    conflict_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Merges two thread entries into:
      - an existing 'left' (if new_id is None and strategy ends with '-inplace-left')
      - an existing 'right' (if new_id is None and strategy ends with '-inplace-right')
      - a new entry (if new_id provided)

    strategy:
      - 'prefer-left' (default): on conflicts, use left value
      - 'prefer-right': on conflicts, use right value
      - 'prefer-left-inplace-left': mutate left in place
      - 'prefer-right-inplace-right': mutate right in place

    conflict_keys: if provided, only these keys are considered conflicting and resolved by strategy.

    Returns the merged thread entry.
    """
    reg = _load_registry()
    Lk, Rk = str(left_id), str(right_id)
    if Lk not in reg or Rk not in reg:
        raise KeyError("One or both thread ids not found.")

    left = deepcopy(reg[Lk])
    right = deepcopy(reg[Rk])

    # Decide merge target
    inplace_left = strategy.endswith("-inplace-left")
    inplace_right = strategy.endswith("-inplace-right")
    create_new = new_id is not None
    if sum([inplace_left, inplace_right, create_new]) != 1:
        # default to create new if nothing selected
        if not any([inplace_left, inplace_right, create_new]):
            create_new = True
            if new_id is None:
                raise ValueError("new_id must be provided for new merge output.")
        else:
            raise ValueError("Ambiguous merge target selection.")

    # Pick base target object
    if create_new:
        if str(new_id) in reg:
            raise ValueError(f"Target id {new_id} already exists.")
        target_id = new_id
        base = {}
        base["id"] = target_id
        # default name
        base["name"] = f"Merged-{int(left_id)}-{int(right_id)}"
    elif inplace_left:
        target_id = left_id
        base = left
    else:
        target_id = right_id
        base = right

    # Define the keys to merge
    keys = set(list(left.keys()) + list(right.keys()))
    # Always preserve immutable ids
    if "id" in keys:
        keys.remove("id")

    # Merge rule
    prefer_left = strategy.startswith("prefer-left")

    def resolve(k, lv, rv):
        if conflict_keys and k not in conflict_keys:
            # If not in conflict scope, default to left then right then base
            return lv if lv is not None else rv
        if lv is None and rv is None:
            return None
        if isinstance(lv, dict) and isinstance(rv, dict):
            # deep merge dicts with preference
            merged = {}
            for dk in set(lv.keys()) | set(rv.keys()):
                if dk in lv and dk in rv:
                    merged[dk] = lv[dk] if prefer_left else rv[dk]
                elif dk in lv:
                    merged[dk] = lv[dk]
                else:
                    merged[dk] = rv[dk]
            return merged
        if isinstance(lv, list) and isinstance(rv, list):
            # concatenate unique, preserve order with preference
            if prefer_left:
                seen = set()
                out = []
                for item in lv + rv:
                    key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
                    if key not in seen:
                        seen.add(key)
                        out.append(item)
                return out
            else:
                seen = set()
                out = []
                for item in rv + lv:
                    key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
                    if key not in seen:
                        seen.add(key)
                        out.append(item)
                return out
        # scalar values
        return lv if prefer_left else rv

    # Perform merge
    merged = deepcopy(base)
    for k in keys:
        lv = left.get(k)
        rv = right.get(k)
        # if only one side has value
        if lv is not None and rv is None:
            merged[k] = lv
        elif rv is not None and lv is None:
            merged[k] = rv
        else:
            merged[k] = resolve(k, lv, rv)

    # Normalize required fields
    merged["id"] = target_id
    if not merged.get("name"):
        merged["name"] = f"Merged-{int(left_id)}-{int(right_id)}"

    # lineage and parent tracking
    lineage = merged.get("lineage", [])
    if left.get("id") not in lineage:
        lineage.append(left.get("id"))
    if right.get("id") not in lineage:
        lineage.append(right.get("id"))
    merged["lineage"] = lineage
    merged["parent"] = merged.get("parent", None)

    # histories
    hist = merged.get("history", [])
    if isinstance(hist, list):
        _append_history(merged, f"Merged threads {left_id} and {right_id} into {target_id} with strategy '{strategy}'.")
    else:
        merged["history"] = []
        _append_history(merged, f"Merged threads {left_id} and {right_id} into {target_id} with strategy '{strategy}'.")

    # Save
    reg[str(target_id)] = merged
    _save_registry(reg)
    logger.info(f"Merged {left_id} + {right_id} -> {target_id} ({strategy})")
    return merged


def fork_thread(
    source_id: float,
    new_ids: List[float],
    per_thread_overrides: Optional[List[Optional[Dict[str, Any]]]] = None
) -> List[Dict[str, Any]]:
    """
    Forks a source thread into multiple new threads.
    Each fork is equivalent to a branch with its own overrides.
    """
    if per_thread_overrides and len(per_thread_overrides) != len(new_ids):
        raise ValueError("per_thread_overrides length must match new_ids length.")

    reg = _load_registry()
    key = str(source_id)
    if key not in reg:
        raise KeyError(f"Source thread {source_id} not found.")

    out = []
    for idx, nid in enumerate(new_ids):
        overrides = per_thread_overrides[idx] if per_thread_overrides else None
        created = branch_thread(source_id=source_id, new_id=nid, overrides=overrides)
        out.append(created)
    logger.info(f"Forked {source_id} into {len(new_ids)} children: {', '.join(map(str, new_ids))}")
    return out


def reparent_thread(child_id: float, new_parent_id: float) -> Dict[str, Any]:
    """
    Changes the parent of a given thread and updates lineage.
    """
    reg = _load_registry()
    ck, pk = str(child_id), str(new_parent_id)
    if ck not in reg or pk not in reg:
        raise KeyError("Child and/or new_parent not found.")

    child = deepcopy(reg[ck])

    child["parent"] = reg[pk]["id"]
    lineage = child.get("lineage", [])
    if reg[pk]["id"] not in lineage:
        lineage.append(reg[pk]["id"])
    child["lineage"] = lineage

    _append_history(child, f"Reparented to {new_parent_id}.")
    reg[ck] = child
    _save_registry(reg)
    logger.info(f"Reparented {child_id} -> parent {new_parent_id}")
    return child


def rename_thread(thread_id: float, new_name: str) -> Dict[str, Any]:
    """
    Renames a thread.
    """
    reg = _load_registry()
    tk = str(thread_id)
    if tk not in reg:
        raise KeyError(f"Thread {thread_id} not found.")
    entry = deepcopy(reg[tk])
    old = entry.get("name")
    entry["name"] = new_name
    _append_history(entry, f"Renamed from '{old}' to '{new_name}'.")
    reg[tk] = entry
    _save_registry(reg)
    logger.info(f"Renamed {thread_id} to '{new_name}'")
    return entry


def mutate_thread(
    thread_id: float,
    updates: Dict[str, Any],
    safe_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Applies updates to a thread entry with optional allowlist of safe fields.
    'id' cannot be updated.
    """
    reg = _load_registry()
    tk = str(thread_id)
    if tk not in reg:
        raise KeyError(f"Thread {thread_id} not found.")
    entry = deepcopy(reg[tk])

    if "id" in updates and updates["id"] != thread_id:
        raise ValueError("Updating 'id' is not supported.")

    if safe_fields is not None:
        updates = {k: v for k, v in updates.items() if k in safe_fields}

    if "metadata" in updates and isinstance(updates["metadata"], dict):
        # deep merge metadata
        entry["metadata"] = {**entry.get("metadata", {}), **updates.pop("metadata")}

    for k, v in updates.items():
        entry[k] = v

    _append_history(entry, f"Mutated fields: {list(updates.keys())}.")
    reg[tk] = entry
    _save_registry(reg)
    logger.info(f"Mutated {thread_id}: {list(updates.keys())}")
    return entry


def summarize_registry() -> Dict[str, Any]:
    """
    Returns a compact summary: counts per stage/device and total.
    """
    reg = _load_registry()
    summary = {
        "total": len(reg),
        "by_stage": {},
        "by_device": {},
    }
    for v in reg.values():
        stage = v.get("stage", "UNKNOWN")
        dev = v.get("device", "UNKNOWN")
        summary["by_stage"][stage] = summary["by_stage"].get(stage, 0) + 1
        summary["by_device"][dev] = summary["by_device"].get(dev, 0) + 1
    return summary