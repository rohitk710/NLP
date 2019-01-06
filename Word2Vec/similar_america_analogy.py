import os,sys
import pickle
import numpy as np

from scipy import spatial

model_path = './models/'

loss_model = 'cross_entropy'
fileName = 'AmericaSimilarityCrossEntropy.txt'
if len(sys.argv) > 1:
    if sys.argv[1] == 'nce':
        loss_model = 'nce'
        fileName = 'AmericaSimilarityNCE.txt'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))
dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

resultfile = open(fileName,'w')

result = ""

vectorfirst = embeddings[dictionary['first']]
vectorAmerican = embeddings[dictionary['american']]
vectorWould = embeddings[dictionary['would']]

similarityFirst = []
similarityAmerican = []
similarityWould = []

embeddingWord = []

resultWould = []
resultAmerican = []
resultFirst = []


for key,val in dictionary.items():
    vectorWordInEmbedding = embeddings[val]
    embeddingWord.append(key)
    similarityFirst.append(1.0 - spatial.distance.cosine(vectorfirst, vectorWordInEmbedding))
    similarityAmerican.append(1.0 - spatial.distance.cosine(vectorAmerican, vectorWordInEmbedding))
    similarityWould.append(1.0 - spatial.distance.cosine(vectorWould, vectorWordInEmbedding))

similarityFirst = sorted(range(len(similarityFirst)), key=lambda i: similarityFirst[i]) [-21:]
similarityAmerican = sorted(range(len(similarityAmerican)), key=lambda i: similarityAmerican[i]) [-21:]
similarityWould = sorted(range(len(similarityWould)), key=lambda i: similarityWould[i]) [-21:]

for i in range(21):
    if(embeddingWord[similarityAmerican[i]] != 'american'):
        result += embeddingWord[similarityAmerican[i]] + ' '
        resultAmerican.append(embeddingWord[similarityAmerican[i]])
result +='\n'

for i in range(21):
    if(embeddingWord[similarityWould[i]] != 'would'):
        result += embeddingWord[similarityWould[i]]+ ' '
        resultWould.append(embeddingWord[similarityWould[i]])
result += '\n'

for i in range(21):
    if(embeddingWord[similarityFirst[i]] != 'first'):
        result += embeddingWord[similarityFirst[i]]+' '
        resultFirst.append(embeddingWord[similarityFirst[i]])
result += '\n'

resultfile.write(result)
resultfile.close()
