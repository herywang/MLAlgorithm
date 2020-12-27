import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = [
    'I love my dog.',
    'I love my cat',
    'Do you love my pig?',
    'Do you think my dog is amazing?'
]
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

test_data = [
    'I really love my dog',
    'my dog very loves my manatee'
]
sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = tokenizer.texts_to_sequences(test_data)
# matrix = tokenizer.texts_to_matrix(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded)
print(word_index)
print(sequences)
# print(matrix)
print('test sequences:', test_sequences)