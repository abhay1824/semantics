import chromadb

# Step 1: Create a persistent client (data saved to ./chroma_db folder)
client = chromadb.PersistentClient(path="./chroma_db")

# Step 2: Create collection with cosine distance
collection = client.get_or_create_collection(
    name="tech_docs",
    metadata={"hnsw:space": "cosine"}
)

# Step 3: Batch ingest 10 IT troubleshooting articles
documents = [
    "Monitor hardware display troubleshooting: Check power cable, VGA/HDMI connection, and brightness settings if screen stays black.",
    "Laptop battery not charging: Inspect charger, power outlet, and battery health in device settings.",
    "Wi-Fi connectivity issues: Restart router, forget and rejoin network, update network drivers.",
    "Blue screen of death (BSOD) fixes: Check recent driver updates, run memory diagnostics, boot in safe mode.",
    "Printer not responding: Verify USB/network connection, restart print spooler service, reinstall drivers.",
    "Slow computer performance: Check startup programs, disk usage, run disk cleanup and defragmentation.",
    "Email client sync errors: Verify IMAP/SMTP settings, check internet connection, clear cache.",
    "Keyboard or mouse not working: Check USB port, replace batteries (wireless), reinstall drivers.",
    "Software installation failures: Run as administrator, check disk space, disable antivirus temporarily.",
    "Overheating laptop: Clean vents and fans, check for dust buildup, use a cooling pad.",
]

categories = [
    "hardware", "hardware", "network", "os",
    "peripherals", "performance", "software",
    "peripherals", "software", "hardware"
]

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    metadatas=[{"category": cat} for cat in categories]
)

# Step 4: Query with natural language
query = "My screen stays black on boot"
results = collection.query(
    query_texts=[query],
    n_results=3
)

# Step 5: Show results with distance
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"Match: {doc}")
    print(f"Category: {meta['category']} | Distance: {dist:.4f}")
    if dist < 0.35:
        print("✅ Strong match")
    print("---")