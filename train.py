import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from config.config import Config
from data.dataset import TranslationDataset
from models.model import build_transformer_model
from validation import run_validation
from tqdm import tqdm


class TranslationTrainer:
    """
    Main trainer class for sequence-to-sequence translation models
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Setting up data loaders and tokenizers...")
        dataset_raw = self._download_dataset()
        self.src_tokenizer = self._get_or_build_tokenizer(dataset_raw, config.lang_src)
        self.tgt_tokenizer = self._get_or_build_tokenizer(dataset_raw, config.lang_tgt)
        
        raw_train_dataset, raw_validation_dataset = self._split_dataset(dataset_raw)
        
        train_dataset = TranslationDataset(
            dataset=raw_train_dataset,
            src_lang=config.lang_src,
            tgt_lang=config.lang_tgt,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            seq_len=config.seq_len
        )
        
        validation_dataset = TranslationDataset(
            dataset=raw_validation_dataset,
            src_lang=config.lang_src,
            tgt_lang=config.lang_tgt,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            seq_len=config.seq_len
        )
        
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=1,
            shuffle=False
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")
        
        print("Setting up model...")
        self.model = build_transformer_model(
            src_vocab_size=self.src_tokenizer.get_vocab_size(),
            tgt_vocab_size=self.tgt_tokenizer.get_vocab_size(),
            src_seq_len=config.seq_len,
            tgt_seq_len=config.seq_len,
            d_model=config.d_model,
            d_ff=config.d_ff
        ).to(self.device)
        
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {param_count:,} trainable parameters")
        
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tgt_tokenizer.token_to_id("<PAD>")
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            eps=1e-9,
            betas=(0.9, 0.98)
        )
        
        self.writer = SummaryWriter(config.experiment_name)

            
    def load_checkpoint(self):
        """
        Load model checkpoint if specified
        """

        initial_epoch = 0
        global_step = 0
        
        if self.config.preload:
            try:
                model_filename = self.config.get_model_path(self.config.preload)
                print(f"Loading checkpoint: {model_filename}")
                
                checkpoint = torch.load(model_filename, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                initial_epoch = checkpoint.get("epoch", 0) + 1
                global_step = checkpoint.get("global_step", 0)
                
                print(f"Resuming from epoch {initial_epoch}, step {global_step}")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch...")
        else:
            print("Starting training from scratch...")
            
        return initial_epoch, global_step
        
    def train_epoch(self, epoch, global_step):
        """
        Train for one epoch
        """

        self.model.train()
        total_loss = 0
        num_batches = 0
        
        batch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch in batch_iterator:
            try:
                encoder_input = batch["encoder_input"].to(self.device, non_blocking=True)
                decoder_input = batch["decoder_input"].to(self.device, non_blocking=True)
                encoder_mask = batch["src_mask"].to(self.device, non_blocking=True)
                decoder_mask = batch["tgt_mask"].to(self.device, non_blocking=True)
                label = batch["label"].to(self.device, non_blocking=True)
                
                model_output = self.model.forward(
                    src=encoder_input,
                    tgt=decoder_input,
                    src_mask=encoder_mask,
                    tgt_mask=decoder_mask
                )
                
                loss = self.loss_fn(
                    model_output.view(-1, self.tgt_tokenizer.get_vocab_size()),
                    label.view(-1)
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if global_step % 100 == 0:
                    self.writer.add_scalar("Loss/train", loss.item(), global_step)
                    
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
                
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch} - Average loss: {avg_loss:.4f}")
        
        return global_step, avg_loss
        
    def validate(self, epoch):
        """
        Run validation
        """

        print(f"Running validation for epoch {epoch}...")
        
        try:
            sources, predictions, targets = run_validation(
                model=self.model,
                validation_dataloader=self.validation_dataloader,
                tokenizer_src=self.src_tokenizer,
                tokenizer_tgt=self.tgt_tokenizer,
                seq_len=self.config.seq_len,
                print_msg=print,
                device=self.device,
                num_examples=3
            )
            
            # Log validation examples to tensorboard
            for i, (src, pred, tgt) in enumerate(zip(sources[:3], predictions[:3], targets[:3])):
                self.writer.add_text(f"Validation/Example_{i}", 
                                   f"Source: {src}\nTarget: {tgt}\nPredicted: {pred}", epoch)
                                   
        except Exception as e:
            print(f"Error during validation: {e}")
            
    def save_checkpoint(self, epoch, global_step):
        """
        Save model checkpoint
        """

        try:
            Path(self.config.model_folder).mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": vars(self.config)  # Save config for reproducibility
            }
            
            model_path = self.config.get_model_path(epoch)
            torch.save(checkpoint, model_path)
            print(f"Model saved: {model_path}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            
    def train(self):
        """
        Main training loop
        """

        print("Starting training...")
        
        # Load checkpoint if specified
        initial_epoch, global_step = self.load_checkpoint()
        
        try:
            for epoch in range(initial_epoch, self.config.num_epochs):
                print(f"Starting epoch {epoch}/{self.config.num_epochs}")
                
                # Train for one epoch
                global_step, avg_loss = self.train_epoch(epoch, global_step)
                
                # Validate every few epochs
                if (epoch + 1) % getattr(self.config, 'validation_frequency', 5) == 0:
                    self.validate(epoch)
                
                # Save checkpoint
                self.save_checkpoint(epoch, global_step)
                
                # Log epoch metrics
                self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
                self.writer.flush()
                
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            if self.writer:
                self.writer.close()
            print("Training completed")
            
    # Helper methods (extracted from original functions)
    def _download_dataset(self):
        """
        Download translation dataset
        """

        return load_dataset(
            "Helsinki-NLP/opus_books",
            f"{self.config.lang_src}-{self.config.lang_tgt}",
            split="train"
        )
        
    def _split_dataset(self, dataset, train_ratio=0.9):
        """
        Split dataset into train/validation
        """

        train_size = int(train_ratio * len(dataset))
        validation_size = len(dataset) - train_size
        return random_split(dataset, [train_size, validation_size])
        
    def _get_or_build_tokenizer(self, dataset, language):
        """
        Get or build tokenizer for specified language
        """

        tokenizer_filename = self.config.get_tokenizer_path(language)
        tokenizer_file_path = Path.cwd() / tokenizer_filename
        
        if not tokenizer_file_path.exists():
            print(f"Building tokenizer for {language}...")
            tokenizer_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordLevelTrainer(
                special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"],
                min_frequency=2
            )
            
            sentence_iter = (item["translation"][language] for item in dataset)
            tokenizer.train_from_iterator(sentence_iter, trainer)
            tokenizer.save(str(tokenizer_file_path))
        else:
            print(f"Loading tokenizer: {tokenizer_file_path}")
            tokenizer = Tokenizer.from_file(str(tokenizer_file_path))
            
        return tokenizer


def main():
    """
    Main entry point
    """

    config = Config()
    
    if config.ignore_warnings:
        warnings.filterwarnings('ignore')
        
    trainer = TranslationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()