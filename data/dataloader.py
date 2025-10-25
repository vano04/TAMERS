import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

MELD_LABELS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
lab2id = {l:i for i,l in enumerate(MELD_LABELS)}

class AudioDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split

        candidate_dirs = [
            self.root / "data" / split,
            self.root / split,
        ]

        self.split_dir = None
        for cand in candidate_dirs:
            meta_candidate = cand / "meta.jsonl"
            if meta_candidate.exists():
                self.split_dir = cand
                break

        if self.split_dir is None:
            locations = ", ".join(str(c / "meta.jsonl") for c in candidate_dirs)
            raise FileNotFoundError(
                f"Could not locate meta.jsonl for split '{split}'. Checked: {locations}."
            )

        meta_file = self.split_dir / "meta.jsonl"
        with open(meta_file, "r", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        audio_path = self.split_dir / item["audio"]
        wav_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # [T, C]
        waveform = torch.from_numpy(wav_np).transpose(0, 1)  # [C, T]
        # mix to mono: [T]
        mono = waveform.mean(dim=0).contiguous()
        text = item["Utterance"]
        label = item["Emotion"].lower()
        # guard against variants like "Sadness" etc.
        if label not in lab2id:
            raise ValueError(f"Unknown label {label}; expected one of {MELD_LABELS}")
        return mono, sr, text, lab2id[label]

def get_dataloader(
    root,
    split="train",
    batch_size=8,
    num_workers=0,
    collate_fn=None,
    shuffle=None,
    pin_memory=None,
    prefetch_factor=None,
):
    ds = AudioDataset(root, split)
    if shuffle is None:
        shuffle = (split == "train")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **loader_kwargs
    )
