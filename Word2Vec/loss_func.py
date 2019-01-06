import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    # Multiplying and then taking diagonal values as the result for all words would be there
    A = tf.diag_part(tf.matmul(inputs, tf.transpose(true_w)))

    #Calculating the denominator.
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, tf.transpose(true_w))),1, keepdims = True))

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    #Converting unigram probability list to tensor
    unigram_prob = tf.convert_to_tensor(unigram_prob)

    #Storing negative words sample words size
    negative_sample_size = sample.shape[0]

    #Batch size
    batch_size = labels.shape[0]

    #Embedding size
    embedding_size = inputs.shape[1]

    #Getting true words weights vector
    true_words = tf.nn.embedding_lookup(weights, labels)
    true_words = tf.reshape(true_words, [-1, embedding_size])

    #Biases for true words
    biases_true_words = tf.nn.embedding_lookup(biases, labels)
    biases_true_words = tf.reshape(biases_true_words, [batch_size])

    #Getting false words weights vector
    false_words = tf.nn.embedding_lookup(weights, sample)

    print false_words.shape

    #Biases for false words
    biases_false_words = tf.nn.embedding_lookup(biases, sample)


    #Getting true words unigram probability
    unigram_true_word_prob = tf.nn.embedding_lookup(unigram_prob, labels)
    unigram_true_word_prob = tf.reshape(unigram_true_word_prob, [batch_size])


    #Getting false words unigram probability
    unigram_false_words_prob = tf.nn.embedding_lookup(unigram_prob, sample)


    # Score function output for true words
    scores_true_words = tf.diag_part(tf.matmul(inputs, tf.transpose(true_words))) + biases_true_words
    
    # Scaled noise distribution for true words
    scaled_noise_true_words = tf.log(tf.scalar_mul(negative_sample_size, unigram_true_word_prob) + 1e-10)

    #Taking logistical value
    A = tf.log(tf.sigmoid(scores_true_words - scaled_noise_true_words) + 1e-10)
    
    #Reshaping Biases for false words to help broadcasting of values
    biases_false_words = tf.reshape(biases_false_words,[-1,1])


    # Score function output for false words
    scores_false_words = tf.transpose(tf.matmul(inputs, tf.transpose(false_words))) + biases_false_words
    
    
    # Scaled noise distribution for false words
    scaled_noise_false_words = tf.scalar_mul(float(negative_sample_size),unigram_false_words_prob)
    scaled_noise_false_words = tf.reshape(scaled_noise_false_words,[-1,1])

    #Taking logistical value
    logistic_value_false_words = tf.sigmoid(tf.subtract(scores_false_words, tf.log(scaled_noise_false_words + 1e-10)))
    
    #Noise Sample Contribution
    B = tf.reduce_sum( tf.log(tf.subtract(1.0, logistic_value_false_words) + 1e-10), 0)
 
    # Final value
    C = -(A + B)

    return C

