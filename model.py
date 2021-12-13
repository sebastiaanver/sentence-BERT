import torch

from transformers import BertModel


class SentenceBert(torch.nn.Module):
    def __init__(self, objective="cosine_similarity", pooling="mean", bert_model=None):
        super(SentenceBert, self).__init__()
        self.bert_layer = bert_model if bert_model else BertModel.from_pretrained('bert-base-uncased')
        self.objective = objective
        self.pooling = pooling
        if self.objective == "cosine_similarity":
            self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif self.objective == "classification":
            self.weights = torch.nn.Linear(in_features=768 * 3, out_features=3)
            self.softmax = torch.nn.Softmax()
        self.double()

    def forward(self, sent_a, sent_b):
        sent_a_out = self.bert_layer(**sent_a)
        if self.pooling == "mean":
            sent_a_pooled = torch.mean(sent_a_out.last_hidden_state, dim=1)
        else:
            sent_a_pooled = sent_a_out.pooler_output

        sent_b_out = self.bert_layer(**sent_b)
        if self.pooling == "mean":
            sent_b_pooled = torch.mean(sent_b_out.last_hidden_state, dim=1)
        else:
            sent_b_pooled = sent_b_out.pooler_output

        if self.objective == "cosine_similarity":
            return self.cos_sim(sent_a_pooled, sent_b_pooled)
        elif self.objective == "classification":
            subs = torch.abs(torch.sub(sent_a_pooled, sent_b_pooled))
            concat = torch.cat((sent_a_pooled, sent_b_pooled, subs), dim=1)
            logits = self.weights(concat)
            return self.softmax(logits)
        else:
            print("Objective not valid")
        return None


class SentenceBertInference:
    def __init__(self, tokenizer, bert_model):
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def predict(self, sentence):
        sentence = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        sentence = self.bert_model(**sentence)
        sentence_embedding = torch.mean(sentence.last_hidden_state, dim=1)

        return sentence_embedding
