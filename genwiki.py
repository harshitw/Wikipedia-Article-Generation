# generate some wikipedia articles using lstms
# coding an lstm from scratch, no keras

# dataset from wikitext long term dependency language modelling dataset
# a bunch of tokenized dataset

# dependencies
import numpy as np # for vectorisation
import random # to generate probablity distribution to generate text
import tensorflow as tf
import datetime # clocking the training time

text = open('wiki.test.raw').read()
print('length of text in no of characters', len(text))

print('head of text:')
print(text[:1000])

# charater by character training and character by character output

# print out our characters and sort them
chars = sorted(list(set(text))) # first remove the duplicate character convert into a list then sort them alpha numerically
char_size = len(char)
print('number of characters', char_size)
print(chars)

# need it to create labels
char2id = dict(c, i) for i,c in enumerate(chars)

id2char = dict(i, c) for i,c in enumerate(chars)

# generate the probablity of each next character
def sample(prediction): # list of possible charaters as input, predicts most likely character
# basicaly a multiclass classification
    r = random.uniform(0, 1)
    # store prediction character
    s = 0
    char_id = len(prediction) - 1
    # for each character prediction probablity
    for i in range(len(prediction)):
        s += prediction[i]
        if s >= r:
            char_id = i
            break

    char_one_hot = np.zeros(shape[char_size])
    char_one_hot[char_id] = 1.0
    return char_one_hot

# vectrize data to feed it into the model

len_per_section = 50 # 50 char long batches will go directly into the training
skip= 2
sections = []
next_chars = []

# hello i am harshit
# llo i am harshit
# o i am harshit
# we will reuse some of the data that is they will overlap
# we don't hacve much data

for i in range(0, len(text), - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])

# Vectorize the input and output
X = np.zeros(len(sections), len_per_section, char_size)
# labels are going to be the next characters for all the character id's, still zero
y = np.zeros(len(sections), char_size)
# for each char in each section, convert each char to an ID
# for each section convert labels into ids
for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i, char2id[next_chars[i]]] = 1

# we are gonna add some VR into it. Machine Learning time!!

batch_size = 512
max_step = 72000
log_every = 100
save_every = 6000
hidden_nodes = 1024
test_start = 'i am thinking that'
# save our model
checkpoint_directory = 'ckpt' # for debugging purposes

# create a checkpoint directory
if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

print('training data size', len(X))
print('approximate steps per epoch:', int(len(X)/batch_size))

# build our model
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0)

    data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
    labels = tf.placeholder(tf.float32, [batch_size, char_size])

    # input gate, output gate, forget gate, internal state
    # they will be calculated in vaccums

    # This is the low level right now. We have neural nets inside of neural nets, neuralceptions basically
    # input gate has weights for inputs, weights for previous output and weights for bias vector
    w_ii = tf.Variable(tf.truncated_normal, ([char_size, hidden_nodes], -0.1, 0.1))
    w_io = tf.Variable(tf.truncated_normal, ([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_i = tf.Varible(tf.zeros([1, hidden_nodes]))

    # forget gate
    w_fi = tf.Variable(tf.truncated_normal, ([char_size, hidden_nodes], -0.1, 0.1))
    w_fo = tf.Variable(tf.truncated_normal, ([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_f = tf.Varible(tf.zeros([1, hidden_nodes]))

    # output gate
    w_oi = tf.Variable(tf.truncated_normal, ([char_size, hidden_nodes], -0.1, 0.1))
    w_oo = tf.Variable(tf.truncated_normal, ([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_o = tf.Varible(tf.zeros([1, hidden_nodes]))

    # memory cell
    w_ci = tf.Variable(tf.truncated_normal, ([char_size, hidden_nodes], -0.1, 0.1))
    w_co = tf.Variable(tf.truncated_normal, ([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_c = tf.Varible(tf.zeros([1, hidden_nodes]))

    def lstm(i, o, state):
        # these are all calculated seperately, no overall until......
        input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)

        forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)

        output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)

        memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

        state = forget_gate * state + input_gate * memory_cell # state right here is the given state

        output = output_gate * tf.tanh(state)

        return output, state

        # LSTM
        # both start of as empty, Lstm will calculate This
        output = tf.zeros([batch_size, hidden_nodes])
        state = tf.zeros([batch_size, hidden_nodes])

        for i in range(len_per_section):
            # calculate state and output from LSTM
            output, state = lstm(data[:, i, :], output_state)
            # to start
            if i == 0:
                # store initial output and labels
                output_all_i = output
                labels_all_i = data[:, i+1, :]
            # for each new set, concat output and labels
            elif i != len_per_section - 1:
            # concatenates (combines) vectors along a dimension axis, not multiply
                output_all_i = tf.concat(0, [outputs_all_i, output])
                labels_all_i = tf.concat(0, [labels_all_i, data[:, i+1, :]])
            else:
                #final score
                outputs_all_i = tf.concat(0, [output_all_i, output])
                labels_all_i = tf.concat(0, [labels_all_i, labels])

        # Classifier
        # last set of weights for the bigger network
        w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size, -0.1, 0.1]))
        b = tf.Variable(tf.zeros([char_size]))

        logits = tf.matmul(output_all_i, w) + b

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_all_i))

        optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step = global_step)

# time to train a model, initialize session with a graph
with tf.Session(graph = graph) as sess:
    tf.global_variables_initializer().run()
    offset = 0
    saver = tf.train.Saver()

    # for each training step
    for step in range(max_steps):
        
