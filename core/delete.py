import os
import json
import sys
import logging
import datetime

# === File Paths ===
THREAD_REGISTRY_PATH = os.path.abspath("./thread_registry.json")
DELETED_BACKUP_PATH = os.path.abspath("./deleted_threads.json")

# === Logger Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeleteThread")


def load_registry(path):
    """Safely loads the thread registry JSON file."""
    if not os.path.exists(path):
        logger.warning("Thread registry not found.")
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("Registry file is corrupted or unreadable.")
        return {}


def save_registry(path, data):
    """Saves updated thread registry to disk."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def backup_deleted_thread(thread_id, thread_data):
    """Backs up a deleted thread's data to a separate archive."""
    archive = {}
    if os.path.exists(DELETED_BACKUP_PATH):
        try:
            with open(DELETED_BACKUP_PATH, "r") as f:
                archive = json.load(f)
        except json.JSONDecodeError:
            archive = {}

    archive[str(thread_id)] = {
        "deleted_at": datetime.datetime.utcnow().isoformat() + "Z",
        "data": thread_data
    }

    with open(DELETED_BACKUP_PATH, "w") as f:
        json.dump(archive, f, indent=2)


def delete_thread_by_id(thread_id):
    """
    Deletes a thread from the thread registry.

    Parameters
    ----------
    thread_id : str or float
        The unique thread identifier to remove.
    """
    data = load_registry(THREAD_REGISTRY_PATH)
    key = str(thread_id)

    if key not in data:
        logger.info(f"No thread found with ID: {thread_id}")
        return

    thread_data = data[key]
    del data[key]
    save_registry(THREAD_REGISTRY_PATH, data)
    backup_deleted_thread(thread_id, thread_data)
    logger.info(f"Thread {thread_id} deleted and archived.")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        logger.error("Usage: python delete.py <thread_id1> <thread_id2> <thread_id3> ...")
        return

    ids = sys.argv[1:]
    for tid in ids:
        delete_thread_by_id(tid)


if __name__ == "__main__":
    main()

# END OF FILE: create.py
# MODULES: The File Itslef
