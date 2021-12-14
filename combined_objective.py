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
    parser.add_argument("--push_to_hub", type=bool, help="If models should be uploaded to huggingface.", default=False)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Fine-tune on the classification task
    train_generator, test_generator = load_data(device, tokenizer, objective="classification")

    model = SentenceBert(objective="classification")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, total_iters=500)
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
            scheduler.step()

            if step % 300 == 0:
                with torch.no_grad():
                    predictions = np.array([])
                    labels = np.array([])
                    for local_batch, local_labels in tqdm.tqdm(test_generator):
                        sent_a, sent_b = local_batch["sent_a"], local_batch["sent_b"]
                        y_pred = model(sent_a, sent_b)
                        y_pred = y_pred.cpu().detach().numpy()
                        y_pred = np.argmax(y_pred, axis=1)
                        predictions = np.append(predictions, y_pred)
                        labels = np.append(labels, local_labels.cpu().detach().numpy())
                    acc = accuracy_score(predictions, labels)
                    print(f"Accuracy of the model: {acc}")

    # Fine-tune on the regression task
    train_generator, test_generator = load_data(device, tokenizer, objective="cosine_similarity")
    model = SentenceBert(objective="cosine_similarity", bert_model=model.bert_layer)

    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, total_iters=150)
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
            scheduler.step()

    if args.push_to_hub:
        model.bert_layer.push_to_hub("sentence-BERT-combined")

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
