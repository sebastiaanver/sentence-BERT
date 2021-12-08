import torch

from transformers import BertModel


class SentenceBert(torch.nn.Module):
    def __init__(self, objective="cosine_similarity"):
        super(SentenceBert, self).__init__()
        self.bert_layer = BertModel.from_pretrained("bert-base-uncased")
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.objective = objective
        # self.double()

    def forward(self, sent_a, sent_b):
        sent_a_out = self.bert_layer(**sent_a)
        sent_a_pooled = torch.mean(sent_a_out.last_hidden_state, dim=1)

        sent_b_out = self.bert_layer(**sent_b)
        sent_b_pooled = torch.mean(sent_b_out.last_hidden_state, dim=1)

        if self.objective == "cosine_similarity":
            return self.cos_sim(sent_a_pooled, sent_b_pooled)

        return None
