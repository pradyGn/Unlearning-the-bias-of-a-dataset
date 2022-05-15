# Unlearning-the-bias-of-a-dataset

Problem Statement: Our problem statement is to identify and explore techniques that would help NLI models be more generalized and improve performance on challenge datasets. 

Approach: There are three approaches by which we have tried to reducing dataset bias.

1. As shown by He et al, learning a biased model by using features that are known to cause dataset bias and then training a debiased model which fits the residuals of the biased model is an effective way to unlearn dataset bias. This approach would provide us with our baseline performance and validate the results of He He, 20191.

2. We have trained the BERT model by modifying the approach proposed by He et al. We have masked words in the premise sentences while MLM pre-training so as to increase the focus of the model on premise sentences and inturn reduce the focus of the model on hypothesis sentences (potential bias causing features).

3. A number of models have been finetuned by masking 15% percetage of words in the hypothesis sentences (every model has a different set of words masked). We have masked these words rather than removing it so that the model retains it's semantic understanding capabilities without giving away the actual meaning of that particular word. The motivation behind this approach is that, each word (in the hypothese) contributes towards the bias and we want to reduce the total contribution towards bias by averaging it.

Models: Bidirectional Encoder Representations from Transformers (BERT) (Devlin, 2019)

Datasets: The two (challenge) datasets which have been used to test the implementation are Stanford Natural Language Inference (SNLI) (Bowman, 2015)2 and Multi-Genre Natural Language Inference (MNLI) (Williams, 2017)3.

The training architecture looks as follows:

![alt text](https://raw.githubusercontent.com/pradyGn/Unlearning-the-bias-of-a-dataset/main/Architecture.png)


The python scripts can be run using the following commands (we highly recommend running it using batch scheduler on HPC). The scripts need to be ran in the following order.

```
python MLMPre-trainingBiasedModel.py
python finetuningBiasedModel.py
python MLMPre-trainingDebiasedModel.py
python finetuningDebiasedModel.py
```
