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

    train_generator, test_generator = load_data(device, tokenizer, objective=args.objective)

    model = SentenceBert(objective=args.objective)
    model.to(device)

    if args.objective == "cosine_similarity":
        criterion = torch.nn.MSELoss(reduction="sum")
    elif args.objective == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    step = 0
    for epoch in range(args.epochs):
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

    torch.save(model.state_dict(), "model")

    with torch.no_grad():
        predictions = np.array([])
        labels = np.array([])
        for local_batch, local_labels in tqdm.tqdm(test_generator):
            sent_a, sent_b = local_batch["sent_a"], local_batch["sent_b"]
            y_pred = model(sent_a, sent_b)
            y_pred = y_pred.cpu().detach().numpy()
            if args.objective == "classification":
                y_pred = np.argmax(y_pred, axis=1)
            predictions = np.append(predictions, y_pred)
            labels = np.append(labels, local_labels.cpu().detach().numpy())
        if args.objective == "cosine_similarity":
            r = stats.spearmanr(predictions, labels)
            print(f"Spearman correlation: {r.correlation}")
        if args.objective == "classification":
            print(predictions.shape)
            acc = accuracy_score(predictions, labels)
            print(f"Accuracy of the model: {acc}")


if __name__ == "__main__":
    main()
