import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, device):

        self.sent_a_tensor = tokenizer(
            list(df["sent_a"]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        ).to(device)
        self.sent_b_tensor = tokenizer(
            list(df["sent_b"]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        ).to(device)
        self.labels = df["scaled_score"].values

    def __getitem__(self, index):
        sent_a_input_ids = torch.tensor(
            self.sent_a_tensor["input_ids"][index]
        ).squeeze()
        sent_b_input_ids = torch.tensor(
            self.sent_b_tensor["input_ids"][index]
        ).squeeze()
        sent_a_attention_masks = torch.tensor(
            self.sent_a_tensor["attention_mask"][index]
        ).squeeze()
        sent_b_attention_masks = torch.tensor(
            self.sent_b_tensor["attention_mask"][index]
        ).squeeze()

        return (
            {
                "sent_a": {
                    "input_ids": sent_a_input_ids,
                    "attention_mask": sent_a_attention_masks,
                },
                "sent_b": {
                    "input_ids": sent_b_input_ids,
                    "attention_mask": sent_b_attention_masks,
                },
            },
            self.labels[index],
        )

    def __len__(self):
        return len(self.labels)


def read_sts_csv(
    path, columns=["source", "type", "year", "id", "score", "sent_a", "sent_b"]
):
    rows = []
    with open(path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for lnr, line in enumerate(lines):
            cols = line.split("\t")
            cols = cols[:7]
            rows.append(cols)
    result = pd.DataFrame(rows, columns=columns)
    result["score_f"] = result["score"].astype("float64")
    result["sent_b"] = result["sent_b"].str.strip()
    return result


def load_data(device, tokenizer):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    train_df = read_sts_csv("data/sts-train.csv")
    test_df = read_sts_csv("data/sts-test.csv")

    train_df["scaled_score"] = scaler.fit_transform(
        np.array(train_df["score"]).reshape(-1, 1)
    )
    test_df["scaled_score"] = scaler.fit_transform(
        np.array(test_df["score"]).reshape(-1, 1)
    )

    params = {
        "batch_size": 16,
        "shuffle": True,
    }
    test_params = {
        "batch_size": 1,
        "shuffle": True,
    }
    train_dataset = Dataset(train_df, tokenizer, device)
    train_generator = torch.utils.data.DataLoader(train_dataset, **params)

    test_dataset = Dataset(test_df, tokenizer, device)
    test_generator = torch.utils.data.DataLoader(test_dataset, **test_params)

    return train_generator, test_generator
