import tensorflow as tf
from tensorflow.keras import Model, losses, layers, Input, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
import numpy as np

from model_builder_config2 import build_discriminator, build_generator, MyGan, generatorCallback

print(f"Version of tensorflow: {tf.__version__}\n")


def main():
    generator = build_generator()
    discriminator = build_discriminator()

    gan = MyGan(generator=generator, disciminator=discriminator)

    generator_loss = losses.BinaryCrossentropy()
    discriminator_loss = losses.BinaryCrossentropy()
    generator_optimizer = Adam(learning_rate=1e-4)
    # generator_optimizer = Adam(learning_rate=0.0002)
    discriminator_optimizer = Adam(learning_rate=1e-3)
    # discriminator_optimizer = Adam(learning_rate=0.0002)


    gan.compile(generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer)
    
    (x_train, _), (x_test, _) = mnist.load_data("/home/bhavsar/raunav/mnist-gan-2"+"/mnist.npz")
    # (x_train, _), (x_test, _) = mnist.load_data()

    all_digits = np.concatenate([x_train, x_test])
    all_digits = all_digits.astype("float32") - 127.5
    all_digits = all_digits / 127.5
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

    batch_size=100
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    tf.random.set_seed(32)
    latent_vectors = tf.random.normal((25, 100))
    set_callbacks = [
        callbacks.CSVLogger(filename="mnist-gan-trianing-config2.csv", separator=",", append=False),
        generatorCallback(latent_vectors, period=20),
        callbacks.ModelCheckpoint(
            filepath="./checkpoints/model-epoch{epoch:02d}.h5",
            save_weights_only=True,
            save_best_only=False, period=50),
    ]
    hist = gan.fit(dataset, epochs=1000, batch_size=batch_size, callbacks=set_callbacks)
    # gan.generator.save(filepath="./generator.h5", overwrite=True, include_optimizer=True)
    # gan.discriminator.save(filepath="./disciminator.h5", overwrite=True, include_optimizer=True)


if __name__ == "__main__":
    main()
