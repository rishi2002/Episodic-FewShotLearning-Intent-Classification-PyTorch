import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np


def batch_from_episode(episode, batch_size):
    """
    Takes input episode and returns support-query pairs along with binary target
    """
    xs = []    
    xq = []
    y = []
    support, query = episode
    for s_idx in range(len(support['texts'])):
        for q_idx in range(len(query['texts'])):
            if support['labels'][s_idx]==query['labels'][q_idx]: 
                bin_label = 1.0
            else:
                bin_label = 0.0
            y.append(bin_label)
            xs.append(support['texts'][s_idx])
            xq.append(query['texts'][q_idx])
    for bidx in range(0,len(xs),batch_size):
        yield xs[bidx:bidx+batch_size], xq[bidx:bidx+batch_size], torch.tensor(y[bidx:bidx+batch_size])

def evaluate(model, num_episodes, batch_size, eval_sampler):
    model.eval()
    accuracies = []
    for ep_idx in range(num_episodes):
        torch.cuda.empty_cache()
        episode = eval_sampler.sample()

        x_support = episode[0]['texts']
        x_query = episode[1]['texts']
        y_support = episode[0]['labels']
        y_query = episode[1]['labels']
        preds = model.predict(x_support, x_query, y_support, batch_size)
        accuracy = accuracy_score(y_query, preds)

        accuracies.append(accuracy)
    return np.mean(accuracies)