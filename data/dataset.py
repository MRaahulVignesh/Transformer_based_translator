from torch.utils.data import Dataset
import torch
from utils.utils import causal_mask

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, seq_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len

        # Fixed: Remove list wrapping for token_to_id calls
        self.special_tokens = {
            "SOS": self.src_tokenizer.token_to_id('<SOS>'),
            "EOS": self.src_tokenizer.token_to_id('<EOS>'),
            "PAD": self.src_tokenizer.token_to_id('<PAD>'),
        }

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]
        
        src_token_ids = self.src_tokenizer.encode(src_text).ids
        tgt_token_ids = self.tgt_tokenizer.encode(tgt_text).ids
        
        src_paddings_tokens = self.seq_len - len(src_token_ids) - 2
        tgt_padding_tokens = self.seq_len - len(tgt_token_ids) - 1

        if src_paddings_tokens < 0 or tgt_padding_tokens < 0:
            raise ValueError("Sentence too long")

        encoder_input = torch.cat([
            torch.tensor([self.special_tokens["SOS"]], dtype=torch.int64),
            torch.tensor(src_token_ids, dtype=torch.int64),
            torch.tensor([self.special_tokens["EOS"]], dtype=torch.int64),
            torch.tensor([self.special_tokens["PAD"]] * src_paddings_tokens, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            torch.tensor([self.special_tokens["SOS"]], dtype=torch.int64),
            torch.tensor(tgt_token_ids, dtype=torch.int64),
            torch.tensor([self.special_tokens["PAD"]] * tgt_padding_tokens, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(tgt_token_ids, dtype=torch.int64),
            torch.tensor([self.special_tokens["EOS"]], dtype=torch.int64),
            torch.tensor([self.special_tokens["PAD"]] * tgt_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len, f"encoder input size {encoder_input.size(0)} not equal to sequence length {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"decoder input size {decoder_input.size(0)} not equal to sequence length {self.seq_len}"
        assert label.size(0) == self.seq_len, f"label size {label.size(0)} not equal to sequence length {self.seq_len}"

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_mask": (encoder_input != self.special_tokens["PAD"]).unsqueeze(0).unsqueeze(0).int(),
            "tgt_mask": (decoder_input != self.special_tokens["PAD"]).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "src_txt": src_text,
            "tgt_txt": tgt_text 
        }