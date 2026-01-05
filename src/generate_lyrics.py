import os
import csv
import whisper
from tqdm import tqdm

AUDIO_DIR = "data/audio"
LYRICS_DIR = "data/lyrics"
METADATA_CSV = "data/metadata.csv"

MAX_FILES = None  # None = generate for all files

os.makedirs(LYRICS_DIR, exist_ok=True)

model = whisper.load_model("base")  # CPU-friendly model

count = 0
skipped = 0

with open(METADATA_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in tqdm(rows, desc="Generating lyrics"):
    track_id = row["track_id"]
    genre = row["genre"]

    audio_path = os.path.join(AUDIO_DIR, genre, f"{track_id}.wav")
    if not os.path.exists(audio_path):
        skipped += 1
        continue

    out_path = os.path.join(LYRICS_DIR, f"{track_id}.txt")
    if os.path.exists(out_path):
        continue

    try:
        result = model.transcribe(audio_path, fp16=False)
        text = (result.get("text") or "").strip()
    except Exception as e:
        print(f"Skipping corrupted file: {audio_path}")
        skipped += 1
        continue

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(text + "\n")

    count += 1
    if MAX_FILES is not None and count >= MAX_FILES:
        break

print(f"Saved {count} lyric files into {LYRICS_DIR}")
print(f"Skipped {skipped} files due to errors")
