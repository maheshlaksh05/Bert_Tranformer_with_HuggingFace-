# Sentiment Analysis of Bert_Tranformer_with_HuggingFace & UsingFine_Tuning_Pretrained_Model_Spam_Mail_Using_Hugging_FaceTransformers
BERT
Overview
The BERT model was proposed in BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It’s a bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia.

The abstract from the paper is the following:

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
![image](https://user-images.githubusercontent.com/97341259/151913706-a9196a04-e24e-4ed3-88a4-fb418a8d95ce.png)


BERT is a model with absolute![image](https://user-images.githubusercontent.com/97341259/151912866-2c93dd5b-1442-4542-b666-2b011999d3c2.png)
 position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.
This model was contributed by thomwolf. The original code can be found here.
Natural language processing (NLP) is one of the most cumbersome areas of artificial intelligence when it comes to data preprocessing. Apart from the preprocessing and tokenizing text datasets, it takes a lot of time to train successful NLP models. But today is your lucky day! We will build a sentiment classifier with a pre-trained NLP model: BERT.
What is BERT?
BERT stands for Bidirectional Encoder Representations from Transformers and it is a state-of-the-art machine learning model used for NLP tasks. Jacob Devlin and his colleagues developed BERT at Google in 2018. Devlin and his colleagues trained the BERT on English Wikipedia (2,500M words) and BooksCorpus (800M words) and achieved the best accuracies for some of the NLP tasks in 2018. There are two pre-trained general BERT variations: The base model is a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, whereas the large model is a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture. Figure 2 shows the visualization of the BERT network created by Devlin et al.

Figure 2. Overall pre-training and fine-tuning procedures for BERT (Figure from the BERT paper)
So, I don’t want to dive deep into BERT since we need a whole different post for that. In fact, I already scheduled a post aimed at comparing rival pre-trained NLP models. But, you will have to wait for a bit.
Additionally, I believe I should mention that although Open AI’s GPT3 outperforms BERT, the limited access to GPT3 forces us to use BERT. But rest assured, BERT is also an excellent NLP model. Here is a basic visual network comparison among rival NLP models: BERT, GPT, and ELMo:

Figure 3. Differences in pre-training model architectures of BERT, GPT, and ELMo (Figure from the BERT paper)
Installing Hugging Face Transformers Library
One of the questions that I had the most difficulty resolving was to figure out where to find the BERT model that I can use with TensorFlow. Finally, I discovered Hugging Face’s Transformers library.
Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, etc in 100+ languages. Its aim is to make cutting-edge NLP easier to use for everyone.
We can easily load a pre-trained BERT from the Transformers library. But, make sure you install it since it is not pre-installed in the Google Colab notebook.
