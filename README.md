# NLP Challenge ecthr-a Dataset

This is an NLP project for predicting violated articles given court case records. I am using the
lex_glue European Court of Human Rights (ECTHR_a)
[dataset](https://huggingface.co/datasets/lex_glue/viewer/ecthr_a/train) from Hugging Face. The
dataset consists of text and labels, where the text is composed of multiple facts that constitute a
court case and the labels represent the violated articles. A single court case can violate more than
one article or none, therefore, this is a multi-label classification problem. Moreover, because we
are given the labels with each example, we can use supervised learning.

The report of my project can be found in the `/docs` directory in the `main.pdf` file. It contains a
basic overview of how I approached the task, the methods I have used and the results I have
obtained.

## Approach 

For this task, I have fine tuned a pre-trained [BERT][1] model on the ECTHRA_a dataset. I used the
BERT Tokeniser for transforming each court case into a 512-long numerical vector.

[1]: <https://arxiv.org/abs/1810.04805>
