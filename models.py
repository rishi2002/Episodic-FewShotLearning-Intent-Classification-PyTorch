from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch

class CrossEncoder(nn.Module):
    def __init__(self, model_name):
        super(CrossEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    def forward(self, x_query, x_support):
        batch = self.tokenizer(x_query, x_support, return_tensors='pt', padding=True)
        batch.to(self.device)
        z = self.encoder(**batch)
        z = z['last_hidden_state'][:,0,:]
        return z

class BinaryClassifier(nn.Module):
    def __init__(self, feat_dim):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(feat_dim, int(feat_dim/2))
        self.linear2 = nn.Linear(int(feat_dim/2), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)        
        x = self.relu(x)        
        x = self.dropout(x)
        scores = self.linear2(x)
        scores = self.sigmoid(scores) 
        return scores
 
class CE_PA(nn.Module):
    def __init__(self, model_name):
        super(CE_PA, self).__init__()
        self.pair_encoder = CrossEncoder(model_name=model_name)
        self.classifier = BinaryClassifier(feat_dim=self.pair_encoder.hidden_size)
    
    def forward(self, x_support, x_query):
        z = self.pair_encoder(x_query=x_query, x_support=x_support)
        y_hat = self.classifier(z)
        return y_hat

    def eval_batchloader(self, x_support, x_query, batch_size):
    
        xs = []    
        xq = []
        
        for s_idx in range(len(x_support)):
            for q_idx in range(len(x_query)):
                xs.append(x_support[s_idx])
                xq.append(x_query[q_idx])
            
        for bidx in range(0,len(xs),batch_size):
            yield xs[bidx:bidx+batch_size], xq[bidx:bidx+batch_size]

    def predict(self, x_support, x_query, y_support, batch_size):
        n_support = len(x_support)
        n_query = len(x_query)
        scores = []
        for n, (x_support, x_query) in enumerate(self.eval_batchloader(x_support, x_query, batch_size)):
            scores.append(self.forward(x_support, x_query))
        # print(scores)
        scores = torch.cat(scores, dim=0)
        scores = scores.view(n_support, n_query)
        max_scores = torch.argmax(scores, dim=0).tolist()
        preds = [y_support[idx] for idx in max_scores]
        return preds