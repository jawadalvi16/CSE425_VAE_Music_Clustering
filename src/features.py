import os
import csv
import numpy as np
import librosa
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

AUDIO_DIR = "data/audio"
LYRICS_DIR = "data/lyrics"
METADATA_CSV = "data/metadata.csv"

# Output files
OUT_AUDIO = "data/audio_mfcc.npy"
OUT_LYRICS = "data/lyrics_emb.npy"
OUT_TRACKS = "data/track_ids.npy"
OUT_GENRES = "data/genres.npy"

# Audio feature settings
SR = 22050              # sample rate
DURATION = 30           # seconds
SAMPLES = SR * DURATION

N_MFCC = 40
MAX_FRAMES = 1300       # fixed time length

MIN_LYRIC_CHARS = 20    # filter super-short/empty transcripts


def load_audio_fixed(path, sr=SR, samples=SAMPLES):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    else:
        y = y[:samples]
    return y


def mfcc_feature(y, sr=SR, n_mfcc=N_MFCC, max_frames=MAX_FRAMES):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc.astype(np.float32)


def main():
    audio_feats = []
    lyrics_embs = []
    track_ids = []
    genres = []

    # Load text embedding model once
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Extracting MFCC + LyricsEmb"):
        track_id = row["track_id"]
        genre = row["genre"]

        # ---- LYRICS CHECK (exists + non-empty) ----
        lyric_path = os.path.join(LYRICS_DIR, f"{track_id}.txt")
        if not os.path.exists(lyric_path):
            continue

        try:
            with open(lyric_path, "r", encoding="utf-8") as lf:
                lyric_text = lf.read().strip()
            if len(lyric_text) < MIN_LYRIC_CHARS:
                continue
        except Exception:
            continue

        # ---- AUDIO CHECK ----
        audio_path = os.path.join(AUDIO_DIR, genre, f"{track_id}.wav")
        if not os.path.exists(audio_path):
            continue

        # ---- FEATURE EXTRACTION ----
        try:
            y = load_audio_fixed(audio_path)
            mfcc = mfcc_feature(y)
        except Exception as e:
            print(f"Skipping audio error: {audio_path} | {e}")
            continue

        try:
            emb = text_model.encode(lyric_text)
        except Exception as e:
            print(f"Skipping text error: {lyric_path} | {e}")
            continue

        audio_feats.append(mfcc)
        lyrics_embs.append(np.array(emb, dtype=np.float32))
        track_ids.append(track_id)
        genres.append(genre)

    audio_feats = np.stack(audio_feats)              # (N, 40, 1300)
    lyrics_embs = np.stack(lyrics_embs)              # (N, 384)
    track_ids = np.array(track_ids)
    genres = np.array(genres)

    np.save(OUT_AUDIO, audio_feats)
    np.save(OUT_LYRICS, lyrics_embs)
    np.save(OUT_TRACKS, track_ids)
    np.save(OUT_GENRES, genres)

    print("Saved:")
    print(f"  {OUT_AUDIO} shape={audio_feats.shape}")
    print(f"  {OUT_LYRICS} shape={lyrics_embs.shape}")
    print(f"  {OUT_TRACKS} shape={track_ids.shape}")
    print(f"  {OUT_GENRES} shape={genres.shape}")


if __name__ == "__main__":
    main()
