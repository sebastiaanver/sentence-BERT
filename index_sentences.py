from datasets import load_dataset
from model import SentenceBertInference
from transformers import BertModel, BertTokenizer

import torch
import tqdm
import argparse
import numpy as np


def index_sentences():
    parser = argparse.ArgumentParser(description="Index sentences")
    parser.add_argument("--model", type=str, default="sentence-BERT-combined")
    args = parser.parse_args()

    model_path = "combined"
    if args.model == "sentence-BERT-regression":
        model_path = "regression"
    elif args.model == "sentence-BERT-classifcation":
        model_path = "classification"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained(f"sebastiaan/{args.model}")

    model = SentenceBertInference(tokenizer, bert_model)

    dataset = load_dataset("Abirate/english_quotes")
    df = dataset["train"].to_pandas()
    df["quote"].apply(lambda x: x.split(".")[0] + '."')

    sentence_embeddings = []
    for quote in tqdm.tqdm(df["quote"]):
        sentence_embeddings.append(model.predict(quote).cpu().detach().numpy()[0])
    df["vec"] = sentence_embeddings

    sentences = df["quote"].values
    vectors = np.vstack(np.ravel(np.array(df["vec"].values)))

    np.save(f"data/{model_path}/sentences.npy", sentences)
    np.save(f"data/{model_path}/vectors.npy", vectors)


if __name__ == "__main__":
    index_sentences()
