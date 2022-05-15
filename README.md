# Unlearning-the-bias-of-a-dataset

###### Problem Statement: 
Our problem statement is to identify and explore techniques that would help NLI models be more generalized and improve performance on challenge datasets. 

###### Approach: 
There are three approaches by which we have tried to reducing dataset bias.

1. As shown by He et al, learning a biased model by using features that are known to cause dataset bias and then training a debiased model which fits the residuals of the biased model is an effective way to unlearn dataset bias. This approach would provide us with our baseline performance and validate the results of He He, 2019.

2. We have trained the BERT model by modifying the approach proposed by He et al. We have masked words in the premise sentences while MLM pre-training so as to increase the focus of the model on premise sentences and inturn reduce the focus of the model on hypothesis sentences (potential bias causing features).

3. A number of models have been finetuned by masking 15% percetage of words in the hypothesis sentences (every model has a different set of words masked). We have masked these words rather than removing it so that the model retains it's semantic understanding capabilities without giving away the actual meaning of that particular word. The motivation behind this approach is that, each word (in the hypothese) contributes towards the bias and we want to reduce the total contribution towards bias by averaging it.

###### Models: 
Bidirectional Encoder Representations from Transformers (BERT) (Devlin, 2019)  

###### Datasets: 
The two datasets which have been used to test the implementation are Stanford Natural Language Inference (SNLI) (Bowman, 2015)2 and Multi-Genre Natural Language Inference (MNLI) (Williams, 2017)3. The challenge dataset useded by He et al is the HANS dataset.  

The Stanford NLI dataset can be accesed through this link: https://nlp.stanford.edu/projects/snli/  
The Multi Genre NLI dataset can be accesed through this link: https://cims.nyu.edu/~sbowman/multinli/  
The HANS dataset can be accesed through this link: https://github.com/hansanon/hans  

###### The training architecture looks as follows:

![alt text](https://raw.githubusercontent.com/pradyGn/Unlearning-the-bias-of-a-dataset/main/Architecture.png)

###### Scripts description:

MLMPre-trainingBiasedModel.py pre-trains a Biased BERT model.    

finetuningBiasedModel.py finetunes the pre-trained Biased BERT model.   

MLMPre-trainingDebiasedModel.py pre-trains a Deiased BERT model.   

finetuningDebiasedModel.py finetunes the pre-trained Debiased BERT model.   

###### Example commands
The python scripts can be run using the following commands (we highly recommend running it using batch scheduler on HPC). The scripts need to be ran in the following order. The implementation shown is for Stanford NLI dataset. However, by simply changing the path of the dataset (to MNLI) at the begining of each file should provide results for respective datasets.

```
python ./MLMPre-trainingBiasedModel.py
python ./finetuningBiasedModel.py
python ./MLMPre-trainingDebiasedModel.py
python ./finetuningDebiasedModel.py
```

For HANS dataset debiased model finetuning, make the suggested changes in the finetuningDebiasedModel.py script (in BertClassifier class). Also, use the commented HANS numerize_y function in the finetuningDebiasedModel.py script script which is commented by default.

###### Results

![image](https://user-images.githubusercontent.com/34334397/168493713-879d46ca-8dea-4566-8e74-caa553513992.png)  

![image](https://user-images.githubusercontent.com/34334397/168493801-83681a98-e5b3-4c99-9a01-33778a64187a.png)  

An ensemble model is also built by masking random sets of words in the hypothesis sentences, and conducting multiple such experiments and averaging the results. The accuracy of this model, along with the other models, is summarized below:

![image](https://user-images.githubusercontent.com/34334397/168493775-78b72f5e-3580-40be-9a4a-c95ff9ad0011.png)  




