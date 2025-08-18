import torch
from inference import translate_text


def run_validation(model, validation_dataloader, tokenizer_src, tokenizer_tgt, seq_len, print_msg, device, num_examples=2):
    """
    Run validation on the model using a validation dataloader.
    """
    source_texts = []
    predicted_texts = []
    target_texts = []
    console_width = 80
    count = 0
    
    for batch in validation_dataloader:
        count += 1
        
        source_text = batch["src_txt"][0]
        target_text = batch["tgt_txt"][0]

        output_text = translate_text(model, tokenizer_src, tokenizer_tgt, source_text, seq_len, device)

        source_texts.append(source_text)
        predicted_texts.append(output_text)
        target_texts.append(target_text)

        if count <= num_examples:
            print_msg('-' * console_width)
            print_msg(f"{'SOURCE:':>12} {source_text}")
            print_msg(f"{'TARGET:':>12} {target_text}")
            print_msg(f"{'PREDICTED:':>12} {output_text}")

        if count == num_examples:
            print_msg('-' * console_width)
            break
    
    return source_texts, predicted_texts, target_texts