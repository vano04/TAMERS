import torch, torch.nn.functional as F
from torch import nn
from transformers import ASTFeatureExtractor, ASTModel, AutoTokenizer, GPT2Model

AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
TXT_ID = "openai-community/gpt2"


class FrozenEncoders(nn.Module):
    """Holds the frozen audio and text encoders used for CLAP fusion."""

    def __init__(self, audio_model_id: str = AST_ID, text_model_id: str = TXT_ID, device: torch.device | None = None) -> None:
        super().__init__()
        runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(audio_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.ast = ASTModel.from_pretrained(audio_model_id, attn_implementation="sdpa").eval()
        self.gpt = GPT2Model.from_pretrained(text_model_id).eval()

        for module in (self.ast, self.gpt):
            for param in module.parameters():
                param.requires_grad = False

        self._device = runtime_device
        self.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, device: torch.device) -> None:
        if device == self._device:
            return
        self._device = device
        self.ast.to(device)  # type: ignore[arg-type]
        self.gpt.to(device)  # type: ignore[arg-type]

    def collate_paired(self, batch):
        """
        batch: [(mono[T], sr:int, text:str, y:int), ...]
        returns:
          a_inputs: dict for AST (pt tensors)
          t_inputs: dict for GPT (pt tensors)
          y: LongTensor [B]
        """
        import numpy as np

        waves_np, texts, ys = [], [], []
        srs = []

        for mono, sr, text, y in batch:
            mono = mono.squeeze().contiguous().cpu()
            waves_np.append(mono.numpy().astype(np.float32))
            srs.append(int(sr))
            texts.append(text)
            ys.append(int(y))

        a_inputs = self.feature_extractor(
            waves_np,
            sampling_rate=srs[0],
            return_tensors="pt",
            padding=True,
        )

        t_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        y = torch.tensor(ys, dtype=torch.long)
        return a_inputs, t_inputs, y

    @torch.no_grad()
    def audio_embed(self, a_inputs):
        x = {k: v.to(self.device, non_blocking=True) for k, v in a_inputs.items()}
        out = self.ast(**x, return_dict=True)
        return F.normalize(out.pooler_output, dim=-1)  # keep CLAP audio vectors unit length

    @torch.no_grad()
    def text_embed(self, t_inputs):
        x = {k: v.to(self.device, non_blocking=True) for k, v in t_inputs.items()}
        out = self.gpt(**x, return_dict=True)
        hidden = out.last_hidden_state
        mask = x["attention_mask"].unsqueeze(-1)
        z = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(z, dim=-1)  # unit-normalize text vectors before fusion


_ENCODERS = FrozenEncoders()


def collate_paired(batch):
    return _ENCODERS.collate_paired(batch)


@torch.no_grad()
def audio_embed(a_inputs):
    return _ENCODERS.audio_embed(a_inputs)


@torch.no_grad()
def text_embed(t_inputs):
    return _ENCODERS.text_embed(t_inputs)


class CLAPFusionHead(nn.Module):
    def __init__(
        self,
        d_ast: int = 768,
        d_txt: int = 768,
        d_shared: int = 1024,
        n_classes: int = 7,
        dropout: float = 0.3,
        hidden_mult: float = 2.0,
        classifier_depth: int = 3,
    ) -> None:
        super().__init__()

        if classifier_depth < 1:
            raise ValueError("classifier_depth must be >= 1")

        hidden_width = max(int(d_shared * hidden_mult), d_shared)

        self.proj_a = self._make_projection(d_ast, d_shared, hidden_width, dropout)
        self.proj_t = self._make_projection(d_txt, d_shared, hidden_width, dropout)

        self.logit_scale = nn.Parameter(torch.tensor(4.6052))  # ~100

        self.clf = self._make_classifier(
            2 * d_shared,
            n_classes,
            hidden_width,
            classifier_depth,
            dropout,
        )

    def forward(self, z_a_raw, z_t_raw):
        z_a = F.normalize(self.proj_a(z_a_raw), dim=-1)
        z_t = F.normalize(self.proj_t(z_t_raw), dim=-1)
        sim = (z_a @ z_t.t()) * self.logit_scale.exp()
        logits = self.clf(torch.cat([z_a, z_t], dim=-1))
        return z_a, z_t, sim, logits

    @staticmethod
    def _make_projection(d_in: int, d_out: int, hidden: int, dropout: float) -> nn.Sequential:
        layers = [
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        ]
        return nn.Sequential(*layers)

    @staticmethod
    def _make_classifier(
        d_in: int,
        d_out: int,
        hidden_width: int,
        depth: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = d_in
        for _ in range(depth):
            layers.extend([nn.Linear(prev, hidden_width), nn.GELU(), nn.Dropout(dropout)])
            prev = hidden_width
        layers.append(nn.Linear(prev, d_out))
        return nn.Sequential(*layers)

def clap_infonce(sim):
    B = sim.size(0)
    y = torch.arange(B, device=sim.device)
    return 0.5*(F.cross_entropy(sim, y) + F.cross_entropy(sim.t(), y))


def set_model_device(new_device) -> None:
    """Move frozen encoder models to the runtime device used by the trainer."""
    _ENCODERS.set_device(new_device)
