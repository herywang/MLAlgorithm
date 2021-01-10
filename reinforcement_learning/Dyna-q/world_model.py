from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras import layers

input_image = Input((64, 64, 3), name='input_state')

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2))(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_image, decoded)
encoder = Model(input_image, encoded)

encoded_input = Input(shape=(32, 32, 32))
decoder = Model(encoded_input, decoded)


