from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    batch_size: int = 8
    num_epochs: int = 20
    lr: float = 1e-4
    seq_len: int = 350
    d_model: int = 512
    d_ff: int = 2048

    lang_src: str = "en"
    lang_tgt: str = "it"

    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: Optional[str] = None
    tokenizer_file: str = "tokenizers/tokenizer_{0}.json"

    experiment_name: str = "runs/tmodel"
    validation_frequency: int = 2
    ignore_warnings: bool = False

    def get_tokenizer_path(self, language: str) -> Path:
        return Path(self.tokenizer_file.format(language))

    def get_model_path(self, epoch = None) -> Path:
        if epoch is None:
            return Path(self.model_folder) / self.model_basename
        return Path(self.model_folder) / f"{self.model_basename}{epoch}.pt"