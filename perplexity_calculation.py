#!/usr/bin/env python3
import tiktoken
import os
import torch
import numpy as np
from contextlib import contextmanager
from model import GPTConfig, GPT

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

class GPTPerplexityAnalyzer:

    dtype = 'bfloat16'
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ctx = torch.no_grad()

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[GPTPerplexityAnalyzer.dtype]
        self.ctx = torch.no_grad()
        self.enc = tiktoken.get_encoding("gpt2")
        self.ends_with = '. '
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = GPTPerplexityAnalyzer.device

    def load_model(self, model_dir):
        ckpt_path = os.path.join(model_dir, 'ckpt.pt')
        print(f"Loading checkpoint from: {ckpt_path}")  # Debug print
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)

        adjusted_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_key = k.replace('_orig_mod.', '')
            adjusted_state_dict[new_key] = v

        model.load_state_dict(adjusted_state_dict, strict=False)
        model.eval()
        model.to(self.device)

        return model

    def get_ppl(self, model, sentence, context):
        encode = lambda s: self.enc.encode(s, allowed_special={""})
        start_ids = encode(context + ' ' + sentence + self.ends_with)
        x = torch.tensor(start_ids[:-1], dtype=torch.long, device=self.device)[None, ...]
        y = torch.tensor(start_ids[1:], dtype=torch.long, device=self.device)[None, ...]

        print(f"Input IDs: {x}")  # Debug print

        with torch.no_grad():
            with self.ctx:
                logits, loss = model(x, y)
                print(f"Logits: {logits}, Loss: {loss}")  # Debug print
                return loss.exp().item()

def calculate_perplexity_for_time_window(root_dir, sentence, time_window):
    """
    Calculate perplexity for a sentence over a specified time window.

    Parameters:
    root_dir (str): The base directory containing model subdirectories.
    sentence (str): The sentence to calculate perplexity for.
    time_window (list): A list of time windows, e.g., ["2015-09", "2015-10"].

    Returns:
    list: A list of perplexity values for each model in the time window.
    """
    analyzer = GPTPerplexityAnalyzer(root_dir)
    perplexities = []

    for time_dir in time_window:
        model_dir = os.path.join(root_dir, time_dir)
        print(f"Checking directory: {model_dir}")  # Debug print
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")

        model = analyzer.load_model(model_dir)
        context = ""  # Provide appropriate context if required

        try:
            perplexity = analyzer.get_ppl(model, sentence, context)
            perplexities.append(perplexity)
        except Exception as e:
            print(f"Error calculating perplexity for sentence: {sentence} in directory {model_dir} - {e}")
            perplexities.append(None)

    return perplexities

