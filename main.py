import torch
import argparse

from dataset import load_data
from model import SentenceBert
from transformers import BertTokenizer


def main():
    parser = argparse.ArgumentParser(description="Sentence BERT")
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training.", default=16
    )
    parser.add_argument("--epochs", type=int, help="Number of train epochs.", default=4)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_generator, test_generator = load_data(device, tokenizer)

    model = SentenceBert()
    model.to(device)

    criterion = torch.nn.MSELoss(reduction="sum")
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


if __name__ == "__main__":
    main()
