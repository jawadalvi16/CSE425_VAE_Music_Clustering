import os
import csv

AUDIO_DIR = "data/audio"
OUTPUT_CSV = "data/metadata.csv"

rows = []

for genre in os.listdir(AUDIO_DIR):
    genre_path = os.path.join(AUDIO_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            track_id = file.replace(".wav", "")
            rows.append([track_id, genre, "en"])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["track_id", "genre", "language"])
    writer.writerows(rows)

print(f"metadata.csv created with {len(rows)} entries")
