#!/bin/python

lines=[]
def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    with open('twitter_output_brown_train.pos') as f:
        for line in f: 
            line = line.strip() #or some other preprocessing
            lines.append(line.split("\t")) #storing everything in memory!

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    
    # New features being added
    #Suffix features
    if word.endswith("ly"):
        ftrs.append("IS_ADVERB")
    if word.endswith("er") or word.endswith("ion") or word.endswith("ist") or word.endswith("ment") or word.endswith("ness") or word.endswith("ity") or word.endswith("ism"):
        ftrs.append("IS_NOUN")
    if word.endswith("less") or word.endswith("ful") or word.endswith("able") or word.endswith("ous") or word.endswith("al") or word.endswith("ish"):
        ftrs.append("IS_ADJECTIVE")
    if word.endswith("ate") or word.endswith("ify") or word.endswith("ise") or word.endswith("ize") or word.endswith("en"):
        ftrs.append("IS_VERB")


    #Prefix Features
    if word.startswith("mis") or word.startswith("dis") or word.startswith("op"):
        ftrs.append("IS_VERB")

    if word.startswith("un"):
        ftrs.append("IS_ADJECTIVE")

    #Other rules
    import string
    if word in string.punctuation:
        ftrs.append("IS_PUNCTUATION")
    if len(word)>1 and word[0].isupper() and word[1].islower():
        ftrs.append("IS_PROPER_NOUN")

    #special Chars
    if "-" in word:
        ftrs.append("HAS_HYPHEN")

    if word.startswith("#") or word.startswith("@") or word is "RT":
        ftrs.append("IS_RANDOM")

    #Brown Clustering
    for sublist in lines:
        if word.lower() == sublist[0]:
            ftrs.append("Cluster"+sublist[1])
            break

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food", "tangential", "Importantly" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
