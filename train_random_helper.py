"""Train the matched random-helper baseline using the notebook implementation."""

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
    parser.add_argument("--code-dir", default=".")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--iterations-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--documents-per-tokenization", type=int, default=50)
    parser.add_argument("--helper-d-model", type=int, default=512)
    parser.add_argument("--student-d-model", type=int, default=1024)
    parser.add_argument("--helper-num-heads", type=int, default=8)
    parser.add_argument("--student-num-heads", type=int, default=8)
    parser.add_argument("--cross-num-heads", type=int, default=8)
    parser.add_argument("--helper-num-layers", type=int, default=6)
    parser.add_argument("--student-num-layers", type=int, default=8)
    parser.add_argument("--helper-theta", type=float, default=20000)
    parser.add_argument("--student-theta", type=float, default=10000)
    parser.add_argument("--cross-theta", type=float, default=15000)
    parser.add_argument("--helper-window", type=int, default=15)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num-cross-blocks", type=int, default=6)
    parser.add_argument("--cross-scale", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--initial-eta", type=float, default=1e-5)
    parser.add_argument("--check-every", type=int, default=50)
    args = parser.parse_args()
    sys.path.insert(0, args.code_dir)
    from common_functionsv13 import AdamWOpt, NewTransformerLM, token_batch_stream, trainingSmallLM

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_args = {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train", "streaming": True}
    enc = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    enc.backend_tokenizer.decoder = decoders.ByteLevel()
    batch_stream = token_batch_stream(get_data_stream(ds_args), enc, args.max_seq_len, args.batch_size, args.documents_per_tokenization)
    newtransLM = NewTransformerLM(enc.vocab_size, args.helper_d_model, args.student_d_model, args.helper_num_heads, args.student_num_heads, args.cross_num_heads, args.helper_num_layers, args.student_num_layers, args.student_theta, args.helper_theta, args.cross_theta, args.max_seq_len, args.helper_window, args.stride, args.num_cross_blocks, device=device).to(device)
    AdamWOptx = AdamWOpt(newtransLM.parameters(), alpha=args.initial_eta)
    loss_history, iter_num = [], args.epochs * args.iterations_per_epoch
    output_file_name = os.path.join(args.output_dir, "checkpointRandomHelper.pt")
    trainingSmallLM(AdamWOptx, newtransLM, batch_stream, iter_num, args.iterations_per_epoch, args.eta_min, args.eta_max, max(1, args.iterations_per_epoch // 10), args.check_every, output_file_name, 100, learning_schedule=True, loss_history=loss_history, init_learn=0, coup=args.cross_scale, device=device)
    print("training has finished. Check if the checkpointRandomHelperFinal.pt is saved and then terminate this cell")
    torch.save({"model": copy.deepcopy(newtransLM.state_dict()), "optimizer": copy.deepcopy(AdamWOptx.state_dict()), "loss_history": list(loss_history)}, os.path.join(args.output_dir, "checkpointRandomHelperFinal.pt"))


if __name__ == "__main__":
    main()
