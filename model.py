import torch

from transformers import BertModel


def pooling_layer(inputs, outputs):
    token_embeddings = outputs[0]
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    pooled = sum_embeddings / sum_mask
    return pooled


class SentenceBert(torch.nn.Module):
    def __init__(self, objective="cosine_similarity", bert_model=None):
        super(SentenceBert, self).__init__()
        self.bert_layer = bert_model if bert_model else BertModel.from_pretrained('bert-base-uncased')
        self.objective = objective
        if self.objective == "cosine_similarity":
            self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif self.objective == "classification":
            self.weights = torch.nn.Linear(in_features=768 * 3, out_features=3)
            self.softmax = torch.nn.Softmax()
        self.double()

    def forward(self, sent_a, sent_b):
        sent_a_out = self.bert_layer(**sent_a, return_dict=False)
        sent_a_pooled = pooling_layer(sent_a, sent_a_out)

        sent_b_out = self.bert_layer(**sent_b, return_dict=False)
        sent_b_pooled = pooling_layer(sent_b, sent_b_out)

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
        sentence_inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        sentence_outputs = self.bert_model(**sentence_inputs)
        sentence_embedding = pooling_layer(sentence_inputs, sentence_outputs)

        return sentence_embedding
