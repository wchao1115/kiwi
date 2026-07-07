# Kiwi

Kiwi is a small PyTorch project for training a simple language model on Thai text and generating new text from a saved checkpoint.

## What it does

- Trains a character-level language model from a text file
- Saves checkpoints during training
- Loads a checkpoint and generates text

## Requirements

- Python 3.10+
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Train the model:

```bash
python main.py --max-iter 5000 --checkpoint-interval 100
```

Generate text with the latest checkpoint:

```bash
python main.py predict
```

Generate text with a specific checkpoint:

```bash
python main.py predict --checkpoint checkpoints/your_checkpoint.chkpt
```

## Project structure

- `main.py` dispatches between training and prediction
- `train.py` trains the model
- `predict.py` loads a checkpoint and generates text
- `dataset.py`, `model.py`, `trainer.py` contain the core implementation
- `books/` contains sample training texts
- `checkpoints/` stores saved model checkpoints

## Notes

The repository intentionally ignores large training artifacts such as checkpoints and downloaded corpus data so a new GitHub repository stays lightweight.
