# BiasInWordEmbedding
## Description
Bias analysis in different word embeddings using hard debias and double hard debias method.
# Getting the Word embedding
The word emebedding we used was modified word2vec google news, here is the [link](https://drive.google.com/file/d/1qDXR4KeH_E0onWNt2sDpd8im6bIgK9br/view?usp=sharing) to download the word embeddings we used.
# Running the Progeam
1. Download word embedding files and place them in a folder called "embedding". 
2. To run the Hard debias method, go to hard_debias folder and run the debias.py
3. To run the Double hard debias method, go to double_debias folder and run double_debias.py
4. To evaluate the result embeddings of hard debias and double hard debias method. Run the evaluation.py under the root folder.
5. To use k-mean clustering to visulize the result word embeddings, run the visulization.py under root folder. 
