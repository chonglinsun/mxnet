from word2vec import *
import numpy as np
from collections import Counter

def readText():
    text = open('data/text8').read().split(' ')
    word_freq = {}
    for word in text:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

    word_array = word_freq.keys()
    word_index = {}
    for index, key in enumerate(word_array):
        word_index[key] = index

    X = map(lambda x: word_index[x], text)
    sample_prob = map(lambda x: word_freq[x] ** 0.75, word_array)
    sum_prob = reduce(lambda x, y: x + y, sample_prob)
    sample_prob = map(lambda x: x / sum_prob, sample_prob)

    for i in xrange(1, len(sample_prob)):
        sample_prob[i] += sample_prob[i - 1]

    return X, word_index, word_array, sample_prob

def negative_sample(sample_prob):
    rand_prob = np.random.rand()
    left, right = 0, len(sample_prob) - 1
    sample_index = -1
    while left <= right:
        mid = left + (right - left) / 2
        if sample_prob[mid] >= sample_prob:
            sample_index = mid
            right = mid - 1
        else:
            left = mid + 1

    return sample_index

np.random.seed(42)
X, word_index, word_array, sample_prob = readText()
num_words = len(word_array)
num_embed = 50
batch_size = 1000
learning_rate = 0.025
update_period = 100000
wd = 0.
momentum = 0.
window = 2
negative = 3
num_epochs = 1

print 'num_words: ', num_words

opt_params = {
    'learning_rate' : learning_rate,
    'wd' : wd,
    'momentum' : momentum,
}

sg = SkipGram(mx.gpu(), 
    input_size=num_words, 
    num_embed=num_embed, 
    batch_size=batch_size, 
    update_period=update_period,
    opt_params=opt_params)

num_tokens = len(X)
#
#
batch = []
num_examples = 0

for i in range(num_tokens):
    middle_word = X[i]
    context_words = X[i - window : i] + X[i + 1 : i + window + 1]
    if (i + 1) % 1000 == 0:
        print 'processed: ', i + 1
    for ctx_word in context_words:
        example = list([middle_word, ctx_word])
        label = 1
        batch.append((example, label))
        num_examples += 1
        if num_examples % batch_size == 0:
            sg.fit(batch)
            batch = []

        for j in range(negative):
            neg_word = negative_sample(sample_prob)
            example = list([middle_word, ctx_word])
            label = 0
            batch.append((example, label))
            num_examples += 1
            if num_examples % batch_size == 0:
                sg.fit(batch)
                batch = []

embedding = sg.get_embedding_cpu()
embedding_dict = dict(zip(word_array, embedding))

np.savez('data/embedding.npz', embedding_dict=embedding_dict)

