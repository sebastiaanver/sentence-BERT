import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, col_names, tokenizer, device):
        self.sent_a_tensor = tokenizer(
            list(df[col_names[0]]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        self.sent_b_tensor = tokenizer(
            list(df[col_names[1]]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        self.labels = df[col_names[2]].values
        self.device = device

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
                    "input_ids": sent_a_input_ids.to(self.device),
                    "attention_mask": sent_a_attention_masks.to(self.device),
                },
                "sent_b": {
                    "input_ids": sent_b_input_ids.to(self.device),
                    "attention_mask": sent_b_attention_masks.to(self.device),
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


def load_data(device, tokenizer, objective, eval=False):
    if objective == "cosine_similarity":
        col_names = ['sent_a', 'sent_b', 'scaled_score']
        scaler = MinMaxScaler(feature_range=(-1, 1))

        train_df = read_sts_csv("data/sts-train.csv")
        test_df = read_sts_csv("data/sts-test.csv")

        # train_df["scaled_score"] = scaler.fit_transform(
        #     np.array(train_df["score"]).reshape(-1, 1)
        # )
        # test_df["scaled_score"] = scaler.fit_transform(
        #     np.array(test_df["score"]).reshape(-1, 1)
        # )

        train_df["scaled_score"] = train_df["score"].apply(lambda x: (float(x) / 2.5) - 1)
        test_df["scaled_score"] = test_df["score"].apply(lambda x: (float(x) / 2.5) - 1)

        params = {
            "batch_size": 16,
            "shuffle": True,
        }
        test_params = {
            "batch_size": 16,
            "shuffle": True,
        }
        if eval:
            test_dataset = Dataset(test_df, col_names, tokenizer, device)
            test_generator = torch.utils.data.DataLoader(test_dataset, **test_params)

            return test_generator

        else:
            train_dataset = Dataset(train_df, col_names, tokenizer, device)
            train_generator = torch.utils.data.DataLoader(train_dataset, **params)

            test_dataset = Dataset(test_df, col_names, tokenizer, device)
            test_generator = torch.utils.data.DataLoader(test_dataset, **test_params)

            return train_generator, test_generator
    elif objective == "classification":
        dataset = load_dataset('snli')
        params = {'batch_size': 16,
                  'shuffle': True,
                  }
        params_test = {'batch_size': 16,
                       'shuffle': True,
                       }

        df_train = dataset['train'].to_pandas()
        df_test = dataset['test'].to_pandas()

        df_train = df_train[df_train['label'] != -1].sample(100000)
        df_test = df_test[df_test['label'] != -1]

        train_dataset = Dataset(df_train, ['premise', 'hypothesis', 'label'], tokenizer, device)
        test_dataset = Dataset(df_test, ['premise', 'hypothesis', 'label'], tokenizer, device)
        train_generator = torch.utils.data.DataLoader(train_dataset, **params)
        test_generator = torch.utils.data.DataLoader(test_dataset, **params_test)

        return train_generator, test_generator
