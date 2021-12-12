import argparse

import numpy as np
import torch
import tqdm
from scipy import stats
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer

from dataset import load_data
from model import SentenceBert


def main():
    parser = argparse.ArgumentParser(description="Sentence BERT")
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training.", default=16
    )
    parser.add_argument("--epochs", type=int, help="Number of train epochs.", default=4)
    parser.add_argument("--objective", type=str, help="Model training objective.", default="cosine_similarity")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Fine-tune on the classification task
    train_generator, _ = load_data(device, tokenizer, objective="classification")

    model = SentenceBert(objective="classification")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    step = 0
    for epoch in range(1):
        for x_batch, y_batch in train_generator:

            sent_a, sent_b = x_batch["sent_a"], x_batch["sent_b"]

            y_pred = model(sent_a, sent_b)
            loss = criterion(y_pred, torch.tensor(y_batch).to(device))

            if step % 100 == 0:
                print(f"Loss at step {step} is {loss}")
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Fine-tune on the regression task
    train_generator, test_generator = load_data(device, tokenizer, objective="cosine_similarity")
    model = SentenceBert(objective="cosine_similarity", bert_model=model.bert_layer)

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    step = 0
    for epoch in range(4):
        for x_batch, y_batch in train_generator:

            sent_a, sent_b = x_batch["sent_a"], x_batch["sent_b"]

            y_pred = model(sent_a, sent_b)
            loss = criterion(y_pred, torch.tensor(y_batch).to(device))

            if step % 100 == 0:
                print(f"Loss at step {step} is {loss}")
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        predictions = np.array([])
        labels = np.array([])
        for local_batch, local_labels in tqdm.tqdm(test_generator):
            sent_a, sent_b = local_batch["sent_a"], local_batch["sent_b"]
            y_pred = model(sent_a, sent_b)
            y_pred = y_pred.cpu().detach().numpy()

            predictions = np.append(predictions, y_pred)
            labels = np.append(labels, local_labels.cpu().detach().numpy())
        r = stats.spearmanr(predictions, labels)
        print(f"Spearman correlation: {r.correlation}")


if __name__ == "__main__":
    main()
