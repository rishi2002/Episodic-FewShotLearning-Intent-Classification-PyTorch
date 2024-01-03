# Few-Shot-Learning-for-Intent-Detection-using-PyTorch
Few shot learning framework for Intent Detection using PyTorch and Huggingface. We use cross-encoders and parametric similarity functions for pairwise matching query samples with support samples. We implement balanced and unbalanced samplers to simulate a real-world setting where each support set class may not have an equal number of samples.
<br>
We used the Banking77 and Clinc150 datasets and split them into train-test-validation splits with mutually exclusive classes. We chose the IntenDD model as our encoder since it is pre-trained on intent detection data. We also implemented a learnable parametric function for similarity scoring. For episodic training, we follow https://aclanthology.org/2023.eacl-main.135.pdf. Since the datasets mentioned above are class-balanced, the approach for unbalanced sampling mentioned in the above paper won't work. Hence, we perform uniform sampling for n-way and k_shot for each class in each episode.
<br>
<h2>Datasets</h2>
I have implemented the code for 
banking77: https://arxiv.org/pdf/2003.04807.pdf 
dataset: https://paperswithcode.com/dataset/banking77
<br>
clinc150: https://arxiv.org/pdf/1909.02027v1.pdf
dataset: https://paperswithcode.com/dataset/clinc150
<br>
The paper also implements the Liu54 and Hwu64 datasets for the balanced setting and ATIS19, SNIPS7 and TOP18 for the unbalanced settings. These datasets have not been implemented or tested in the code.
![image2](https://github.com/rishi2002/Episodic-FewShotLearning-Intent-Classification-PyTorch/assets/74735335/ecc61629-65b8-48b7-956a-e15bccb8336e)
The above image shows the number of episodes and classes in each dataset split.
<h2>Model Components</h2>
<h4>Cross Encoder</h4>
![image](https://github.com/rishi2002/Episodic-FewShotLearning-Intent-Classification-PyTorch/assets/74735335/90fa68eb-baff-45e3-8bd3-fcccf623ca7d)
<h4>Parametric Similarity Scoring</h4>
![image](https://github.com/rishi2002/Episodic-FewShotLearning-Intent-Classification-PyTorch/assets/74735335/7656223c-39cc-42ea-a48f-e2f597a82f97)
<h4>Episodic Training</h4>
![image](https://github.com/rishi2002/Episodic-FewShotLearning-Intent-Classification-PyTorch/assets/74735335/d5f4583a-6583-4a04-a9eb-d75f2e91515b)

<h2>Samplers</h2>
<h4>Balanced Samplers</h4>
The authors follow https://aclanthology.org/2021.acl-long.191.pdf for balanced sampling
![image](https://github.com/rishi2002/Episodic-FewShotLearning-Intent-Classification-PyTorch/assets/74735335/d2349ab3-91ed-4500-8878-c8cba1ad9618)
<h4>Unbalanced Samplers</h4>
The authors follow https://aclanthology.org/2020.nlp4convai-1.12.pdf for unbalanced sampling
refer to section 6.1 of the above paper for detailed unbalanced episode construction
NOTE: since I have implemented the code only on the balanced datasets, the unbalanced sampler does not follow the procedure mentioned in the paper. Sampling n-way and k-shot for each episode can be changed in the samplers.py file. Comments indicate which lines can be changed for the new implementation.

<br><br>
I referenced the code from https://github.com/UKPLab/eacl2023-few-shot-intent-classification

```
python main.py
```
ToDo
[] Add links to files, checkpoints and other resources
[] Add requirements.txt
