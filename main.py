from tensorflow import keras as k
from matplotlib import pyplot as plt
import random


# DATASET
dataset_path = "/home/obergam/Data/flir/images_thermal_train"
image_size = (82, 82)
color_mode = "grayscale"
batch_size = 10
validation_split = 0.2
seed = random.randint(0, 10000)
print(f"Seed: {seed}")

train_dataset = k.preprocessing.image_dataset_from_directory(
        dataset_path,
        labels=None,
        seed=seed,
        image_size=image_size,
        crop_to_aspect_ratio=True,
        batch_size=batch_size,
        color_mode=color_mode
)
train_dataset = train_dataset.map(
        lambda x: (x, x)
)

# MODEL
model = k.Sequential()
model.add(k.Input(shape=image_size + (1,)))
model.add(k.layers.Rescaling(1/255))
for i in range(20):
    model.add(k.layers.Conv2D(64, 3, padding="same"))
    model.add(k.layers.Activation("relu"))
model.add(k.layers.Conv2D(1, 3, padding="same"))
model.summary()

# COMPILATION
lr_schedule = k.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10,
        decay_rate=0.9
)
model.compile(
        optimizer=k.optimizers.Adam(learning_rate=lr_schedule),
        loss=k.losses.MeanSquaredError(),
        metrics=['accuracy']
)

# TRAINING
n_epochs = 10
history = model.fit(
        train_dataset,
        epochs=n_epochs,
        #validation_split=validation_split,
        workers=8
)

# VISUALIZING TRAINING
# Visualize the accuracy of the model
plt.figure(1, figsize=(14, 4))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# Visualize the loss of the model
plt.figure(2, figsize=(14, 4))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.show()
