# Helper-Model Transformers for Language Modeling

This project investigates whether a small, specialized language model can accelerate the training of a larger model with a complementary attention pattern. A **helper model** learns short-range linguistic structure with sliding-window attention, while a **student model** uses dilated attention to model longer-range dependencies. Cross-attention blocks allow the student to retrieve representations from the helper during training and inference.

Unlike conventional knowledge distillation, this method does not train the student to imitate the helper's output probabilities. Instead, it transfers learned hidden representations directly through the architecture.

## Research question

Can a pretrained local-attention model provide useful phrase-level features to a long-range student model, thereby improving the student's learning efficiency?

The experiments compare two otherwise matched composite models:

1. **Pretrained helper:** the helper is pretrained on Cosmopedia before joint training.
2. **Random helper:** the helper begins from a random initialization and is trained jointly with the student.

Both composite models are trained on FineWeb-Edu. Their next-token training losses and generated samples are then compared.

## Architecture

The implementation contains three principal components:

- **Helper stream:** a smaller Transformer using causal sliding-window attention to learn local patterns.
- **Student stream:** a larger Transformer using causal dilated attention to capture long-range context.
- **Cross-attention:** the student supplies queries, while the helper supplies keys and values. The resulting cross-attention output is added to the student stream through residual connections.

The notebook uses the following default configuration:

| Component | Configuration |
|---|---|
| Helper model | 6 layers, model dimension 512, 8 heads |
| Helper attention | Sliding window of 15 tokens |
| Student model | 8 layers, model dimension 1024, 8 heads |
| Student attention | Dilated attention with stride 8 |
| Cross-attention | 6 blocks, 8 heads |
| Sequence length | 256 tokens |
| Batch size | 20 sequences |
| Tokenizer | Local 20k-token Cosmopedia tokenizer |

## Experimental workflow

The provided notebook reproduces the main computational pipeline:

1. Pretrain the helper model on Cosmopedia.
2. Train a student with the pretrained helper on FineWeb-Edu.
3. Train the same student architecture with a randomly initialized helper.
4. Load the saved checkpoints and compare smoothed training-loss trajectories.
5. Generate text from the helper and both composite models.

The current experiments suggest that helper pretraining improves loss reduction during an early portion of training. The advantage later diminishes as the larger student develops stronger representations of its own. This repository should therefore be viewed as an exploratory architectural study rather than evidence of a general improvement in asymptotic language-model performance.

## Repository requirements

Set `code_dir` to the project directory containing the tokenizer and all required source files:

```text
common_functionsv13.py
pretrain_helper.py
train_pretrained_helper.py
train_random_helper.py
reproduce_figure_3.py
cosmopedia-tokenizer-20k/
```

The tokenizer configuration is defined by the accompanying project files. The serialized `cosmopedia-tokenizer-20k/` directory is therefore a required dependency and must be stored inside `code_dir` before training or decoding. It is not downloaded automatically by the notebook. Set `output_dir` to the directory where checkpoints should be written and loaded.

It also creates or loads these checkpoints:

```text
checkpoints/checkpointPhraseFinal.pt
checkpoints/checkpointPretrainedHelperFinal.pt
checkpoints/checkpointRandomHelperFinal.pt
```

## Running the notebook

1. Place the source files and `cosmopedia-tokenizer-20k/` directory together in the project directory.
2. Set `code_dir` to that directory.
3. Set `tokenizer` to `code_dir/cosmopedia-tokenizer-20k` and choose an `output_dir` for checkpoints.
4. Run the notebook cells in order to pretrain the helper, train both composite models, reproduce the loss comparison, and inspect generated samples.

Training is configured for nine learning-schedule cycles of 10,000 iterations each. Reduce `epochs`, `iterations_per_epoch`, or the model dimensions for a shorter test run.

## Main dependencies

The code uses:

- Python 3
- PyTorch
- Hugging Face `datasets`
- Hugging Face `transformers`
- Hugging Face `tokenizers`
- NumPy
- Matplotlib

## Citation context

The project is related to knowledge distillation, sparse attention, and modular representation transfer, but differs from standard output-level distillation by incorporating a pretrained helper's internal features through cross-attention.

## Status

This is a research prototype. The notebook uses configurable path variables for the project directory and checkpoint directory. It relies on companion Python modules, the local tokenizer, and saved checkpoints. Future work includes controlled validation experiments, compute-matched baselines, ablations of the cross-attention placement and scale, and tests at larger sequence lengths.
