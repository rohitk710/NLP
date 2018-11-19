import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            # Setting trainable to  false.
            # self.embeddings = tf.Variable(embedding_array, dtype=tf.float32, trainable= False)
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
           
            # Placeholder for train_inputs of size batch_size * n_Tokens
            # Placeholder for train_labels of size batch_size * numTrans
            # Placeholder for test_inputs of size n_Tokens
            self.train_inputs = tf.placeholder(tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(tf.float32, shape=(Config.batch_size, parsing_system.numTransitions()))
            self.test_inputs = tf.placeholder(tf.int32, shape=(Config.n_Tokens))

            # look up for embedding and reshape to batch_size * (n_tokens * embedding_size)
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, Config.n_Tokens * Config.embedding_size])

            # Variable to store Bias of size hidden_size with intial values set to zero
            # biases_input = tf.Variable(tf.zeros([Config.batch_size]))
            biases_input = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))
            # biases_input = tf.Variable(tf.random_normal([Config.hidden_size], mean=0.0, stddev=0.1))

            # Bisases for second and third hidden layers
            # biases_input_second_layer = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))
            # biases_input_third_layer = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))
            
            # Bisases for three paralle hidden layers
            # biases_input_w = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))
            # biases_input_t = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))
            # biases_input_l = tf.Variable(tf.random_uniform([Config.hidden_size],-1,1))


            # Variable to store input weight for hidden layer of size embedding_size*n_Tokens 
            # with intial values set random values from a normal distribution.
            weights_input = tf.Variable(tf.random_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size], mean=0.0, stddev=0.1))
            
            # Weights for second and third hidden layers
            # weights_input_second_layer = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=0.0, stddev=0.1))
            # weights_input_third_layer = tf.Variable(tf.random_normal([Config.hidden_size, Config.hidden_size], mean=0.0, stddev=0.1))

            # Weights for three paralle hidden layers
            # weights_input_w = tf.Variable(tf.random_normal([Config.embedding_size * Config.w_Tokens, Config.hidden_size], mean=0.0, stddev=0.1))
            # weights_input_t = tf.Variable(tf.random_normal([Config.embedding_size * Config.t_Tokens, Config.hidden_size], mean=0.0, stddev=0.1))
            # weights_input_l = tf.Variable(tf.random_normal([Config.embedding_size * Config.l_Tokens, Config.hidden_size], mean=0.0, stddev=0.1))

            # Variable to store output weight for hidden layer of size hidden_size
            # with intial values set random values from a normal distribution.
            weights_output = tf.Variable(tf.random_normal([Config.hidden_size, parsing_system.numTransitions()], mean=0.0, stddev=0.1))

            #Call forward_pass and get predictions
            self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_output)

            # Forward pass expetiment scenario for two layers.
            # self.prediction = self.forward_pass_two_layers(train_embed, weights_input, weights_input_second_layer, biases_input, biases_input_second_layer, weights_output)
            
            # Forward pass expetiment scenario for three layers.
            # self.prediction = self.forward_pass_three_layers(train_embed, weights_input, weights_input_second_layer, weights_input_third_layer, biases_input, biases_input_second_layer, biases_input_third_layer, weights_output)
            
            # Forward pass expetiment scenario for three parallel hidden layers.
            # self.prediction = self.forward_pass_parallel_hidden(train_embed, weights_input_w, weights_input_t, weights_input_l, biases_input_w, biases_input_t, biases_input_l, weights_output)


            # Cross Entropy loss calculation
            # Give it a batch of predictions and a batch of labels, it will return the loss for each prediction
            # and then we take average across those values
            # tf.nn.relu to avoid -1 values. Computes rectified linear: max(features, 0). 
            # tf.nn.softmax_cross_entropy_with_logits_v2 : Computes softmax cross entropy between logits and labels.
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=tf.nn.relu(self.train_labels)))

            # the theta term is the set of all parameters {Ww1, Wt1, Wl1, b1, W2, Ew, Et, El}. 
            # tf.nn.l2_loss : Computes half the L2 norm of a tensor without the sqrt
            
            normalization_term = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_output))
            
            # Noraml calculation for various experimentation scenarios.
            # normalization_term = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(biases_input_second_layer) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_input_second_layer) + tf.nn.l2_loss(weights_output))
            # normalization_term = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(biases_input_second_layer) + tf.nn.l2_loss(biases_input_third_layer) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_input_second_layer) + tf.nn.l2_loss(weights_input_third_layer) + tf.nn.l2_loss(weights_output))
            # normalization_term = Config.lam * (tf.nn.l2_loss(self.embeddings) + tf.nn.l2_loss(biases_input_w) + tf.nn.l2_loss(biases_input_t) + tf.nn.l2_loss(biases_input_l) + tf.nn.l2_loss(weights_input_w) + tf.nn.l2_loss(weights_input_t) + tf.nn.l2_loss(weights_input_l) + tf.nn.l2_loss(weights_output))

            #Loss Functions
            self.loss = cross_entropy_loss + normalization_term

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)
            #Gradient clipping
            # self.app = optimizer.apply_gradients(grads)


            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # self.test_pred = self.forward_pass_two_layers(test_embed, weights_input, weights_input_second_layer, biases_input, biases_input_second_layer, weights_output)
            # self.test_pred = self.forward_pass_three_layers(test_embed, weights_input, weights_input_second_layer, weights_input_third_layer, biases_input, biases_input_second_layer, biases_input_third_layer, weights_output)
            # self.test_pred = self.forward_pass_parallel_hidden(test_embed, weights_input_w, weights_input_t, weights_input_l, biases_input_w, biases_input_t, biases_input_l, weights_output)
            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """

        hidden_units = tf.add(tf.matmul(embed, weights_input), biases_inpu)
        input_to_hidden = tf.pow(hidden_units,3)
        #input_to_hidden = tf.tanh(hidden_units)
        #input_to_hidden = tf.sigmoid(hidden_units)
        #input_to_hidden = tf.nn.relu(hidden_units)


        hidden_to_output = (tf.matmul(input_to_hidden, weights_output))
        return hidden_to_output

    #Function for two hidden layers
    # def forward_pass_two_layers(self, embed, weights_input, weights_input_second_layer, biases_input, biases_input_second_layer, weights_output):
    #     """
    #     =======================================================

    #     Implement the forwrad pass described in
    #     "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    #     =======================================================
    #     """

    #     hidden_units_1 = tf.add(tf.matmul(embed, weights_input),biases_input)
    #     input_to_hidden1 = tf.pow(hidden_units_1,3)

    #     hidden_units_2 = tf.add(tf.matmul(input_to_hidden1, weights_input_second_layer), biases_input_second_layer)
    #     #input_to_hidden2 = tf.pow(hidden_units_2,3)
    #     input_to_hidden2 = tf.tanh(hidden_units_2)

    #     hidden2_to_output = (tf.matmul(input_to_hidden2, weights_output))
    #     return hidden2_to_output

    # Function for three hidden layers
    # def forward_pass_three_layers(self, embed, weights_input, weights_input_second_layer, weights_input_third_layer, biases_input, biases_input_second_layer, biases_input_third_layer, weights_output):
    #     """
    #     =======================================================

    #     Implement the forwrad pass described in
    #     "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    #     =======================================================
    #     """

    #     hidden_units_1 = tf.add(tf.matmul(embed, weights_input), biases_input)
    #     input_to_hidden1 = tf.pow(hidden_units_1,3)

    #     hidden_units_2 = tf.add(tf.matmul(input_to_hidden1, weights_input_second_layer), biases_input_second_layer)
    #     #input_to_hidden2 = tf.pow(hidden_units_2,3)
    #     input_to_hidden2 = tf.tanh(hidden_units_2)

    #     hidden_units_3 = tf.add(tf.matmul(input_to_hidden2, weights_input_third_layer), biases_input_third_layer)
    #     #input_to_hidden3 = tf.pow(hidden_units_3,3)
    #     input_to_hidden3 = tf.tanh(hidden_units_3)

    #     hidden3_to_output = (tf.matmul(input_to_hidden3, weights_output))
    #     return hidden3_to_output

    # def forward_pass_parallel_hidden(self, embed, weights_input_w, weights_input_t, weights_input_l, biases_input_w, biases_input_t, biases_input_l, weights_output):
    #     """
    #     =======================================================

    #     Implement the forwrad pass described in
    #     "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    #     =======================================================
    #     """
    #     # separating embeddings of words, tags and labels

    #     word_index = Config.embedding_size * Config.w_Tokens
    #     tag_index = word_index + (Config.embedding_size * Config.t_Tokens)
    #     label_index = tag_index + (Config.embedding_size * Config.l_Tokens)
    #     embed_words = embed[:, 0:word_index]
    #     embed_tags = embed[:, word_index:tag_index]
    #     embed_labels = embed[:, tag_index:label_index]
    

    #     hidden_units_w = tf.add(tf.matmul(embed_words, weights_input_w), biases_input_w)
    #     input_to_hidden_w = tf.pow(hidden_units_w,3)

    #     hidden_units_t = tf.add(tf.matmul(embed_tags, weights_input_t), biases_input_t)
    #     input_to_hidden_t = tf.pow(hidden_units_t,3)

    #     hidden_units_l = tf.add(tf.matmul(embed_labels, weights_input_l), biases_input_l)
    #     input_to_hidden_l = tf.pow(hidden_units_l,3)

    #     feed_to_output = tf.add(tf.add(input_to_hidden_w, input_to_hidden_t), input_to_hidden_l)

    #     hidden_to_output = (tf.matmul(feed_to_output, weights_output))
    #     return hidden_to_output


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    #Three lists to store the 18 words, 18 tags and 12 labels
    set_words = []
    set_tags = []
    set_labels = []

    # Adding to the respective lists the top 3 words on the stack and buffer
    # and the corresponding POS tags
    for itr in range(0,3):
        stackWordIndex = c.getStack(itr)
        set_words.append(getWordID(c.getWord(stackWordIndex)))
        set_tags.append(getPosID(c.getPOS(stackWordIndex)))

        bufferWordIndex = c.getBuffer(itr)
        set_words.append(getWordID(c.getWord(bufferWordIndex)))
        set_tags.append(getPosID(c.getPOS(bufferWordIndex)))

    # Adding to the respective lists the first and second leftmost / rightmost children of the 
    # top two words on the stack.
    # and the corresponding POS tags and labels
    for itr in range(0,2):
        left_child1 = c.getLeftChild(c.getStack(itr),1)
        left_child2 = c.getLeftChild(c.getStack(itr),2)

        set_words.append(getWordID(c.getWord(left_child1)))
        set_words.append(getWordID(c.getWord(left_child2)))

        set_tags.append(getPosID(c.getPOS(left_child1)))
        set_tags.append(getPosID(c.getPOS(left_child2)))

        set_labels.append(getLabelID(c.getLabel(left_child1)))
        set_labels.append(getLabelID(c.getLabel(left_child2)))

        right_child1 = c.getRightChild(c.getStack(itr),1)
        right_child2 = c.getRightChild(c.getStack(itr),2)

        set_words.append(getWordID(c.getWord(right_child1)))
        set_words.append(getWordID(c.getWord(right_child2)))
        
        set_tags.append(getPosID(c.getPOS(right_child1)))
        set_tags.append(getPosID(c.getPOS(right_child2)))
        
        set_labels.append(getLabelID(c.getLabel(right_child1)))
        set_labels.append(getLabelID(c.getLabel(right_child2)))

        # Adding to the respective lists the leftmost of leftmost / rightmost of rightmost
        # children of the top two words on the stack:
        # and the corresponding POS tags and labels
    for itr in range(0,2):
        left_left_child = c.getLeftChild(c.getLeftChild(c.getStack(itr),1),1)

        set_words.append(getWordID(c.getWord(left_left_child)))
        set_tags.append(getPosID(c.getPOS(left_left_child)))
        set_labels.append(getLabelID(c.getLabel(left_left_child)))

        right_right_child = c.getRightChild(c.getRightChild(c.getStack(itr),1),1)

        set_words.append(getWordID(c.getWord(right_right_child)))
        set_tags.append(getPosID(c.getPOS(right_right_child)))
        set_labels.append(getLabelID(c.getLabel(right_right_child)))

    #The list to store the above computed features clubbed together
    feature = []
    feature += set_words
    feature += set_tags
    feature += set_labels

    return feature

def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)

    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

