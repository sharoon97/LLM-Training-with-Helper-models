"""Pretrain the sliding-window helper model (NotebookFinal-style driver)."""

import argparse
import copy
import os
import sys
import json
import torch
from datasets import load_dataset
from tokenizers import decoders
from transformers import AutoTokenizer


def get_data_stream(ds_args):
    data_set = load_dataset(**ds_args).shuffle(seed=42, buffer_size=10_000)
    return iter(data_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", default=".", help="Directory containing common_functionsv13.py")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iterations-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--documents-per-tokenization", type=int, default=50)
    parser.add_argument("--helper-d-model", type=int, default=512)
    parser.add_argument("--helper-num-heads", type=int, default=8)
    parser.add_argument("--helper-num-layers", type=int, default=6)
    parser.add_argument("--helper-theta", type=float, default=20000)
    parser.add_argument("--helper-window", type=int, default=15)
    parser.add_argument("--eta-max", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--initial-eta", type=float, default=1e-5)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--ds_args", type= dict, default={"path": "HuggingFaceTB/cosmopedia", "name": "stanford", "split": "train", "streaming": True})
    args = parser.parse_args()

    sys.path.insert(0, args.code_dir)
    from common_functionsv13 import AdamWOpt, PhraseTransformerLM, token_batch_stream, trainingStandardLM
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter = get_data_stream(args.ds_args)
    enc = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    enc.backend_tokenizer.decoder = decoders.ByteLevel()

    batch_stream = token_batch_stream(train_iter, enc, args.max_seq_len, args.batch_size, args.documents_per_tokenization)
    PhraseLM = PhraseTransformerLM(enc.vocab_size, args.helper_d_model, args.helper_num_heads, args.helper_num_layers, args.helper_theta, args.helper_window, device=device).to(device)
    iter_num = args.epochs * args.iterations_per_epoch
    AdamWOptx = AdamWOpt(PhraseLM.parameters(), alpha=args.initial_eta)
    loss_history = []
    output_file_name = os.path.join(args.output_dir, "checkpointPhrase")
    trainingStandardLM(AdamWOptx, PhraseLM, batch_stream, iter_num, args.iterations_per_epoch, args.eta_min, args.eta_max, max(1, args.iterations_per_epoch // 10), args.check_every, output_file_name, 100, learning_schedule=True, loss_history=loss_history, init_learn=0, device=device)
    print("training has finished. Check if the checkpointPhraseFinal.pt is saved and then terminate this cell")
    torch.save({"model": copy.deepcopy(PhraseLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)}, os.path.join(args.output_dir, "checkpointPhraseFinal.pt"))


if __name__ == "__main__":
    main()
