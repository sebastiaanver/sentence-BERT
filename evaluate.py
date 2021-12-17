import torch
import tqdm
import numpy as np
import argparse

from scipy import stats
from dataset import load_data
from model import SentenceBert
from transformers import BertTokenizer, BertModel


def main():
    parser = argparse.ArgumentParser(description="Sentence BERT")
    parser.add_argument("--model", type=str, help="Model.", default="combined")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.model == "combined":
        hf_name = "sebastiaan/sentence-BERT-combined"
    elif args.model == "classification":
        hf_name = "sebastiaan/sentence-BERT-classification"
    elif args.model == "regression":
        hf_name = "sebastiaan/sentence-BERT-regression"
    bert_model = BertModel.from_pretrained(hf_name)

    _, test_generator = load_data(device, tokenizer, objective="cosine_similarity")

    model = SentenceBert(bert_model=bert_model)
    model.to(device)

    with torch.no_grad():
        predictions = np.array([])
        labels = np.array([])
        for local_batch, local_labels in tqdm.tqdm(test_generator):
            sent_a, sent_b = local_batch["sent_a"], local_batch["sent_b"]
            y_pred = model(sent_a, sent_b)
            predictions = np.append(predictions, y_pred.cpu().detach().numpy())
            labels = np.append(labels, local_labels.cpu().detach().numpy())

        r = stats.spearmanr(predictions, labels)
        print(f"Spearman correlation: {r.correlation}")


if __name__ == "__main__":
    main()
