# Few-Shot-Learning-for-Intent-Detection-using-PyTorch
Few shot learning framework for Intent Detection using PyTorch and Huggingface. We use cross-encoders and parametric similarity functions for pairwise matching query samples with support samples. We implement balanced and unbalanced samplers to simulate a real-world setting where each support set class may not have an equal number of samples.
We used the Banking77 and Clinc150 datasets and split them into train-test-validation splits with mutually exclusive classes. We chose the IntenDD model as our encoder since it is pre-trained on intent detection data. We also implemented a learnable parametric function for similarity scoring. For episodic training, we follow https://aclanthology.org/2023.eacl-main.pdf. Since the datasets mentioned above are class-balanced, the approach for unbalanced sampling mentioned in the above paper won't work. Hence, we perform uniform sampling for n-way and k_shot for each class in each episode.

The chosen dataset, balanced/unbalanced settings, hyperparameters, and encoder model can be changed in main.py

python main.py
