import random

class BalancedSampler:
    def __init__(self, full_dataset, class_list, n_way, k_shot):
        self.class_list = class_list
        self.n_way = n_way
        self.k_shot = k_shot
        
        dataset = {}
        for label in class_list:
            dataset[label] = full_dataset[label]
        self.dataset = dataset

    def sample(self):
        support = {'texts':[],
                   'labels':[]}
        query = {'texts':[],
                 'labels':[]}
        n_classes = random.sample(self.class_list, self.n_way)
        for label in n_classes:
            k_samples = random.sample(self.dataset[label],2*self.k_shot)
            random.shuffle(k_samples)
            texts_s = k_samples[:self.k_shot]
            texts_q = k_samples[self.k_shot:]
            support['texts'].extend(texts_s)
            support['labels'].extend([label]*self.k_shot)
            query['texts'].extend(texts_q)
            query['labels'].extend([label]*self.k_shot)
        
        support_final = {}
        query_final = {}
        
        indices = list(range(len(support['texts'])))
        random.shuffle(indices)
        support_final['texts'] = [support['texts'] for i in indices]
        support_final['labels'] = [support['labels'] for i in indices]

        indices = list(range(len(query['texts'])))
        random.shuffle(indices)
        query_final['texts'] = [query['texts'] for i in indices]
        query_final['labels'] = [query['labels'] for i in indices]


        return support, query


class UnbalancedSampler:
    def __init__(self, full_dataset, class_list, n_way, k_shot):
        self.class_list = class_list
        
        dataset = {}
        for label in class_list:
            dataset[label] = full_dataset[label]
        self.dataset = dataset

    def get_n_way(self):
        n_way = random.randint(2, 5) #change this line to implement different sampling for n_way
        return n_way
    
    def get_k_shot(self, sampled_classes):
        k_shot = {}
        for label in sampled_classes:
            k_shot[label] = random.randint(1, 5) #change this line to imlpement different sampling for n_way
        return k_shot
    
    def sample(self):
        n_way = self.get_n_way()
        sampled_classes = random.sample(self.class_list, n_way)
        support_k_shot = self.get_k_shot(sampled_classes)
        query_k_shot = 5
        
        support = {'texts':[],
                   'labels':[]}
        query = {'texts':[],
                 'labels':[]}
        
        for label in sampled_classes:
            k_samples = random.sample(self.dataset[label],query_k_shot+support_k_shot[label])
            random.shuffle(k_samples)
            texts_s = k_samples[:support_k_shot[label]]
            texts_q = k_samples[support_k_shot[label]:]
            support['texts'].extend(texts_s)
            support['labels'].extend([label]*support_k_shot[label])
            query['texts'].extend(texts_q)
            query['labels'].extend([label]*query_k_shot)
        
        support_final = {}
        query_final = {}
        
        indices = list(range(len(support['texts'])))
        random.shuffle(indices)
        support_final['texts'] = [support['texts'] for i in indices]
        support_final['labels'] = [support['labels'] for i in indices]

        indices = list(range(len(query['texts'])))
        random.shuffle(indices)
        query_final['texts'] = [query['texts'] for i in indices]
        query_final['labels'] = [query['labels'] for i in indices]


        return support, query