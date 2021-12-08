import torch
import tqdm
import numpy as np
import argparse

from scipy import stats
from dataset import load_data
from model import SentenceBert
from transformers import BertTokenizer


def main():
    parser = argparse.ArgumentParser(description="Sentence BERT")
    parser.add_argument(
        "--model_path", type=str, help="Model path.", default="/"
    )
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    _, test_generator = load_data(device, tokenizer)
    with torch.no_grad():
        model = SentenceBert()
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)

        predictions = np.array([])
        labels = np.array([])
        for local_batch, local_labels in tqdm.tqdm(test_generator):
            sent_a, sent_b = local_batch["sent_a"], local_batch["sent_b"]
            y_pred = model(sent_a, sent_b)
            predictions = np.append(predictions, y_pred.cpu().detach().numpy())
            labels = np.append(labels, local_labels.cpu().detach().numpy())

        r = stats.spearmanr(predictions, labels)
        print(f"Spearman correlation: {r.correlation}")
        np.save("results.npy", np.array([r.correlation]))


if __name__ == "__main__":
    main()
