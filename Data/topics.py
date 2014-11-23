#! /usr/bin/python

# usage: python topics.py <beta file> <vocab file> <num words>
#
# <beta file> is output from the lda-c code
# <vocab file> is a list of words, one per line
# <num words> is the number of words to print from each topic

import sys, math

def getWeightExpectation(weight, topic):
    for i in range(len(weight)) :
        weight[i] *= math.exp(topic[i])
    return weight

def getEntropy(topic):
    ent = 0
    for t in topic :
        ent += t * math.exp(t)
    return -ent
        
def mean_dev(data):
    mean = 0
    dev = 0
    for item in data:
        mean += item
    mean /= len(data)
    
    for item in data:
        dev += (mean - item) * (mean - item)
    dev = math.sqrt( dev/len(data) )
    if dev == 0 :
        dev = 1e-5
    return (mean, dev)
        
def print_aspects(beta_file, vocab_file, nwords = 25):
    # get the vocabulary
    vocab0 = file(vocab_file, 'r').readlines()
    vocab0 = map(lambda x: x.strip(), vocab0)
    vocab = ['BIAS'] # to include the bias term
    vocab.extend(vocab0)
    wordSize = len(vocab)
    
    # for each line in the beta file
    indices = range(len(vocab))
    
    topic_no = 0
    for aspect in file(beta_file, 'r'):
        aspect = map(float, aspect.strip().split())
        if len(aspect) != wordSize :
            continue
        
        indices.sort(lambda x,y: -cmp(aspect[x], aspect[y]))
        print 'Aspect %d' % topic_no
        for i in range(nwords):
            print '\t' + vocab[indices[i]] + '\t' + str(aspect[indices[i]])                 
        print '----------------------'        
        for i in range(nwords):
            print '\t' + vocab[indices[wordSize-i-1]] + '\t' + str(aspect[indices[wordSize-i-1]])
        topic_no = topic_no + 1
        print '\n'

if (__name__ == '__main__'):
    #print_aspects('e:/Projects/Java/ReviewMiner/Data/Model/model_base_hotel.dat', 'e:/Projects/Java/ReviewMiner/Data/Seeds/hotel_vocabulary_DF.dat', 25)
    if (len(sys.argv) == 4):
        beta_file = sys.argv[1]
        vocab_file = sys.argv[2]
        nwords = int(sys.argv[3])
        print_aspects(beta_file, vocab_file, nwords)
    else:
        print 'usage: python topics.py <beta-file> <vocab-file> <num words>\n'
        sys.exit(1)

    
