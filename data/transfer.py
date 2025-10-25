#!/usr/bin/env python3
"""
transfer.py  — audible-only build
Inputs CSV columns: Utterance, Emotion, Dialogue_ID, Utterance_ID
Media files named: dia{Dialogue_ID}_utt{Utterance_ID}.mp4
Outputs:
  outdir/audio/000001.wav ...
  outdir/meta.jsonl   (only audible items)
"""

import os, csv, json, argparse, subprocess, shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf  # pip install soundfile

# ---------- helpers ----------

def zpad(n: int, width: int = 6) -> str:
    return f"{n:0{width}d}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _run(cmd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)

def probe_has_audio(mp4: Path) -> bool:
    """Return True if file has ANY audio stream (ffprobe)."""
    if not shutil.which("ffprobe"):
        # If ffprobe missing, let conversion + RMS decide later
        return True
    cmd = [
        "ffprobe","-v","error",
        "-select_streams","a","-show_entries","stream=index",
        "-of","csv=p=0", str(mp4)
    ]
    p = _run(cmd)
    return p.returncode == 0 and (p.stdout or "").strip() != ""

def ffmpeg_to_wav(mp4: Path, wav: Path, sr: int) -> Tuple[bool, Optional[str]]:
    """Convert MP4 → mono WAV @ sr. Return (ok, err)."""
    ensure_dir(wav.parent)
    cmd = ["ffmpeg", "-y", "-i", str(mp4), "-ac", "1", "-ar", str(sr), str(wav)]
    p = _run(cmd)
    if p.returncode == 0:
        return True, None
    tail = (p.stderr or p.stdout or "").strip()[-1200:]
    return False, f"ffmpeg failed for {mp4.name}\n{tail}"

def is_near_silent_sf(wav_path: Path, rms_thresh: float, min_active_ratio: float) -> Tuple[bool, float, float]:
    """
    Silence check using soundfile (no torchaudio/torchcodec).
    Returns (silent, rms, active_ratio).
      - rms: root-mean-square over mono signal in [-1,1]
      - active_ratio: fraction of samples with |x| > 5*rms_thresh
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if data.size == 0:
        return True, 0.0, 0.0
    mono = data.mean(axis=1)  # average channels to mono
    rms = float(np.sqrt(np.mean(mono**2)))
    active = int(np.count_nonzero(np.abs(mono) > (5.0 * rms_thresh)))
    active_ratio = float(active) / float(mono.size)
    silent = (rms < rms_thresh) or (active_ratio < min_active_ratio)
    return silent, rms, active_ratio

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--media_root", required=True)
    ap.add_argument("--outdir", default="dataset")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--start_idx", type=int, default=1)
    ap.add_argument("--id_width", type=int, default=6)
    ap.add_argument("--rms_thresh", type=float, default=1e-4,
                    help="RMS below this is considered silent")
    ap.add_argument("--min_active_ratio", type=float, default=0.01,
                    help="Fraction of samples > 5*rms_thresh; below => silent")
    ap.add_argument("--resume", action="store_true", help="Skip existing WAVs that already passed QC")
    args = ap.parse_args()

    media_root = Path(args.media_root)
    outdir = Path(args.outdir)
    audio_dir = outdir / "audio"
    ensure_dir(audio_dir)

    # Load minimal columns
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"Utterance","Emotion","Dialogue_ID","Utterance_ID"}
        miss = need - set(reader.fieldnames or [])
        if miss:
            raise ValueError(f"CSV missing columns: {miss}")
        for r in reader:
            rows.append({
                "Utterance": r["Utterance"],
                "Emotion": r["Emotion"],
                "Dialogue_ID": str(r["Dialogue_ID"]).strip(),
                "Utterance_ID": str(r["Utterance_ID"]).strip(),
            })

    def to_int(x):
        try: return int(x)
        except: return x
    rows.sort(key=lambda r: (to_int(r["Dialogue_ID"]), to_int(r["Utterance_ID"])))

    kept, skipped_no_stream, skipped_silent, failed = 0, 0, 0, 0
    idx = args.start_idx

    meta_path = outdir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as fout:
        for r in rows:
            mp4 = media_root / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4"
            if not mp4.exists():
                # tolerate variants
                cands = list(media_root.glob(f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}*.mp4"))
                if not cands:
                    print(f"[WARN] missing file: {mp4.name}")
                    failed += 1
                    continue
                mp4 = cands[0]

            # Skip if no audio stream at all
            if not probe_has_audio(mp4):
                skipped_no_stream += 1
                continue

            item_id = zpad(idx, args.id_width)
            wav_rel = f"audio/{item_id}.wav"
            wav_abs = outdir / wav_rel

            # Resume support: if file exists, re-check QC; else convert
            if args.resume and wav_abs.exists() and wav_abs.stat().st_size > 0:
                silent, rms, ar = is_near_silent_sf(wav_abs, args.rms_thresh, args.min_active_ratio)
                if silent:
                    skipped_silent += 1
                    # optional: os.remove(wav_abs)
                    continue
            else:
                ok, err = ffmpeg_to_wav(mp4, wav_abs, sr=args.sr)
                if not ok:
                    print(f"[WARN] {err}")
                    failed += 1
                    continue

            # Silence QC
            silent, rms, ar = is_near_silent_sf(wav_abs, args.rms_thresh, args.min_active_ratio)
            if silent:
                skipped_silent += 1
                try:
                    os.remove(wav_abs)
                except Exception:
                    pass
                continue

            # Keep
            fout.write(json.dumps({
                "id": item_id,
                "audio": wav_rel,
                "Utterance": r["Utterance"],
                "Emotion": r["Emotion"],
                "Dialogue_ID": r["Dialogue_ID"],
                "Utterance_ID": r["Utterance_ID"],
            }, ensure_ascii=False) + "\n")
            kept += 1
            idx += 1

    print(f"Wrote {meta_path} (audible only)")
    print(f"WAV in {audio_dir}")
    print(f"Kept: {kept} | Skipped(no-audio stream): {skipped_no_stream} | Skipped(silent): {skipped_silent} | Failed: {failed}")

if __name__ == "__main__":
    main()