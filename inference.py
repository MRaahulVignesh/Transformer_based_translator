import torch
from utils.utils import causal_mask


def greedy_decode(model, tokenizer_tgt, encoder_input, encoder_mask, seq_len, device):
    """
    Perform greedy decoding for sequence-to-sequence translation.
    """
    sos_index = tokenizer_tgt.token_to_id("<SOS>")
    eos_index = tokenizer_tgt.token_to_id("<EOS>")


    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_index).type_as(encoder_input).to(device)

    while decoder_input.size(1) < seq_len:

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        
        probs = model.project(decoder_output[:, -1])
        _, next_word = torch.max(probs, dim=1)
        
        decoder_input = torch.cat([decoder_input, next_word.view(1, 1)], dim=1)

        if next_word.item() == eos_index:
            break

    return decoder_input.squeeze(0)


def translate_text(model, tokenizer_src, tokenizer_tgt, source_text, seq_len, device):
    """
    Translate a single text using the model.
    """
    model.eval()
    
    with torch.no_grad():

        src_tokens = tokenizer_src.encode(source_text).ids
        src_tensor = torch.tensor([src_tokens], dtype=torch.int64).to(device)
        
        src_mask = torch.ones(1, 1, len(src_tokens)).to(device)
        
        model_output = greedy_decode(model, tokenizer_tgt, src_tensor, src_mask, seq_len, device)
        
        try:
            output_tokens = model_output.detach().cpu().tolist()
            
            if output_tokens[0] == tokenizer_tgt.token_to_id("<SOS>"):
                output_tokens = output_tokens[1:]
            
            try:
                eos_idx = output_tokens.index(tokenizer_tgt.token_to_id("<EOS>"))
                output_tokens = output_tokens[:eos_idx]
            except ValueError:
                pass
            
            translated_text = tokenizer_tgt.decode(output_tokens)
            return translated_text
            
        except Exception as e:
            return f"[TRANSLATION ERROR: {str(e)}]"