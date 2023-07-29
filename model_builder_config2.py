import tensorflow as tf
from tensorflow.keras import Model, losses, layers, Input, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np

def build_generator():
    noise = Input(shape=(100,)) #nvis in goodfeli, latent sapce dimention
    
    h0 = layers.Dense(units=1200, name="h0")(noise)
    h0 = layers.BatchNormalization()(h0)
    h0_activated = layers.LeakyReLU(alpha=0.2)(h0)

    h1 = layers.Dense(units=1200, name="h1")(h0_activated)
    h1 = layers.BatchNormalization()(h1)
    h1_activated = layers.LeakyReLU(alpha=0.2)(h1)

    # h2 = layers.Dense(units=1024, name="h2")(h1_activated)
    # h2 = layers.BatchNormalization()(h2)
    # h2_activated = layers.LeakyReLU(alpha=0.2)(h2)

    y = layers.Dense(units=784, activation='tanh', name="y")(h1_activated)
    gen_img = layers.Reshape((28,28,1),name="Gen-img")(y)

    return Model(inputs=[noise,],
                 outputs=[gen_img,],
                 name="Generator")

def build_discriminator():
    img = Input(shape=(28,28,1))
    flat_img = layers.Flatten()(img)

    h0 = layers.Dense(units=240, name="h0")(flat_img)
    h0 = layers.Dropout(0.3)(h0)
    h0_activated = layers.LeakyReLU(alpha=0.2)(h0)

    h1 = layers.Dense(units=240, name="h1")(h0_activated)
    h1 = layers.Dropout(0.3)(h1)
    h1_activated = layers.LeakyReLU(alpha=0.2)(h1)

    # h2 = layers.Dropout(0.8)(h1_activated)

    y = layers.Dense(units=1, activation='sigmoid', name="y")(h1_activated)

    return Model(inputs=[img,],
                 outputs=[y,],
                name="Discriminator")

class MyGan(Model):
    def __init__(self, generator, disciminator, batch_size=100, latent_dim=100):
        super().__init__()
        self.generator = generator
        self.discriminator = disciminator
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.build((1,100))
        # self.summary()
        # # see model description
        # print("GAN components\n\n")
        # self.generator.summary()
        # print("\n\n")
        # self.discriminator.summary()

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
    
    def train_step(self, real_images):
        # latent vector
        latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))

        # Decode and generate image
        generated_images = self.generator(latent_vectors)

        all_images = tf.concat([real_images, generated_images], axis=0)

        #Assemble labels(reals are labeled 1, fakes are labeled 0)
        labels = tf.concat(
            [tf.ones((tf.shape(real_images)[0], 1)),
             tf.zeros((self.batch_size, 1)),], axis=0
        )
        # labels += tf.random.uniform((tf.shape(labels)), minval=-0.05, maxval=0.05) # help the generator a little bit. Restricts the discriminator to learn too fast. Lets the generator to catch up

        # Add random noise to the labels - important trick!
        labels += tf.random.normal(tf.shape(labels))
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_images)
            d_loss = self.discriminator_loss(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Sample random points in the latent space
        latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((self.batch_size, 1))
        
        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(latent_vectors))
            g_loss = self.generator_loss(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"Discriminator loss":d_loss, "Generator loss":g_loss}
    
    # def train_step(self, batch):
    #     # Sample random points in the latent space
    #     latent_dim = 100
    #     batch_size = self.batch_size
    #     real_images = batch
    #     dloss_fn = self.discriminator_loss
    #     gloss_fn = self.generator_loss
    #     d_optimizer = self.discriminator_optimizer
    #     g_optimizer = self.generator_optimizer
    #     # print(type(real_images))
        
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    #     # Decode them to fake images
    #     generated_images = self.generator(random_latent_vectors)
    #     # Combine them with real images
    #     combined_images = tf.concat([generated_images, real_images], axis=0)
    #     # Assemble labels discriminating real from fake images
    #     labels = tf.concat(
    #         [tf.ones((batch_size, 1)), tf.zeros((tf.shape(real_images)[0], 1))], axis=0
    #     )
    #     # Add random noise to the labels - important trick!
    #     # labels += 0.05 * tf.random.uniform(tf.shape(labels))
    #     # Train the discriminator
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator(combined_images)
    #         d_loss = dloss_fn(labels, predictions)
    #     grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
    #     # Sample random points in the latent space
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    #     # Assemble labels that say "all real images"
    #     misleading_labels = tf.zeros((batch_size, 1))
        
    #     # Train the generator (note that we should *not* update the weights
    #     # of the discriminator)!
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator(self.generator(random_latent_vectors))
    #         g_loss = gloss_fn(misleading_labels, predictions)
    #     grads = tape.gradient(g_loss, self.generator.trainable_weights)
    #     g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
    #     return {"Discriminator loss":d_loss, "Generator loss":g_loss}
    
    def call(self, latent_vectors):
        x = self.generator(latent_vectors)
        return self.discriminator(x)
    
    def summary(self):
        super().summary()
        print("\n\n")
        self.generator.summary()
        print("\n\n")
        self.discriminator.summary()
        print("\n\n")

class generatorCallback(callbacks.Callback):
    def __init__(self, latent_vectors, period=10):
        super().__init__()
        self.latent_vectors = latent_vectors
        self.period = period
    def on_train_begin(self, logs=None):
        plt.figure()
        batch_size=25

        generated_images = self.model.generator(self.latent_vectors)
        for i in range(batch_size):
            plt.subplot(5,5,i+1)
            plt.imshow(generated_images[i], cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"./progress_images/epoch-{-1}.png")
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.period == 0:
            plt.figure()
            batch_size=25

            generated_images = self.model.generator(self.latent_vectors)
            for i in range(batch_size):
                plt.subplot(5,5,i+1)
                plt.imshow(generated_images[i], cmap="gray")
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"./progress_images/epoch-{epoch}.png")
            plt.close()
        

