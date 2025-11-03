from core.create import CreateThread , generate_token

token = generate_token()

thread = CreateThread(
    thread_id=1.0,
    stage="CORE",
    device="cpu",
    token=token,
    caching=True,
    priority=2,
    ttl=15.0,
    metadata={"max_memory_mb": 1000},
    thread_name="YourCoreThread"
)
