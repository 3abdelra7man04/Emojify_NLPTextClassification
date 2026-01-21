import emoji
import numpy as np
import tensorflow as tf

## label_emoji takes a numeric label and convert it to the required emoji
def label_emoji(label):
    labels = {0:"\u2764\uFE0F", 1:":baseball:", 2:":smiling_face_with_smiling_eyes:", 3:":pensive_face:", 4:":fork_and_knife:"}
    return emoji.emojize(labels[label])

## read_glove_vecs reads the embeddings file and returns a word_to_vec_map
def read_glove_vec(file):
    with open(file, "r", encoding = "utf-8") as f:
        lines = f.readlines()
    
    word_to_vec_map = {}
    word_to_index = {}
    index_to_word = {}

    for idx in range(len(lines)):
        line = list(lines[idx].strip().split())
        key = line[0]
        values = [float(i) for i in line[1:]]

        ## map each word to its vector
        word_to_vec_map[key] = np.array(values)

        ## map each word to its index in the vocabulary
        word_to_index[key] = idx

        ## map each idx to its word
        index_to_word[idx] = key
    
    return word_to_index, index_to_word, word_to_vec_map

## convert each label to a one hot vector of (,5) dimension
def convert_to_oh(labels, num_classes = 5):
    return tf.one_hot(labels, depth = num_classes)