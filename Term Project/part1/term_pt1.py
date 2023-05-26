from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
from skimage.util import random_noise
import copy
import numpy as np
import random
from skimage.transform import rotate

# important constants
num_classes = 10
img_shape = (28,28,1)


def display_image(img, title=None, gray:bool = True):
    if gray:
        im_show = plt.imshow(img, cmap='gray') 
    else:
        im_show = plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    
# import and normalize
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


for i in range(5):
  display_image(x_train[i], "image label: " + str(y_train[i]))
  
  
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model1 = keras.Sequential([
    keras.Input(shape=img_shape),
    layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation="softmax"),
])
model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
start_time = time.perf_counter()
history = model1.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test,y_test))
end_time = time.perf_counter()

test_time = end_time - start_time

# plot training loss, validation loss 
plt.plot(history.history['loss'], color='red', label='training loss') 
plt.plot(history.history['val_loss'], color='blue', label='validation loss') 
plt.title("Training Loss, Validation Loss - Model 1")
plt.legend(loc="center right")
plt.show()
# plot training accuracy, validation accuracy
plt.plot(history.history['accuracy'], color='red', label='training accuracy') 
plt.plot(history.history['val_accuracy'], color='blue', label='validation accuracy') 
plt.title("Training Accuracy, Validation Accuracy - Model 1")
plt.legend(loc="center right")
plt.show()

# report training time
print(f"Training time: {round(test_time,1)}")

# evaluate the model and report loss and accuracy 
eval_score = model1.evaluate(x_test, y_test, verbose=0)
print(f"Eval loss: {round(eval_score[0],3)}, Eval accuracy: {round(eval_score[1],3)}")



# Part B

model2 = keras.Sequential([
    keras.Input(shape=img_shape),
    layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation="softmax"),
])
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
start_time = time.perf_counter()
history = model2.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test,y_test))
end_time = time.perf_counter()

test_time = end_time - start_time

# plot training loss, validation loss 
plt.plot(history.history['loss'], color='red', label='training loss') 
plt.plot(history.history['val_loss'], color='blue', label='validation loss') 
plt.title("Training Loss, Validation Loss - Model 2")
plt.legend(loc="center right")
plt.show()
# plot training accuracy, validation accuracy
plt.plot(history.history['accuracy'], color='red', label='training accuracy') 
plt.plot(history.history['val_accuracy'], color='blue', label='validation accuracy') 
plt.title("Training Accuracy, Validation Accuracy - Model 2")
plt.legend(loc="center right")
plt.show()

# report training time
print(f"Training time: {round(test_time, 1)}")

# evaluate the model and report loss and accuracy 
eval_score = model2.evaluate(x_test, y_test, verbose=0)
print(f"Eval loss: {round(eval_score[0],3)}, Eval accuracy: {round(eval_score[1],3)}")



# Part C

droupout_accuracies = {}
for rate in [0.1, 0.25, 0.5, 0.75, 0.9]:
  model_c = keras.Sequential([
    keras.Input(shape=img_shape),
    layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(rate),  
    layers.Dense(32, activation='relu'),
    layers.Dropout(rate),  
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation="softmax"),
  ]) 

  model_c.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  history = model_c.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test,y_test))

  droupout_accuracies[rate] = model_c.evaluate(x_test, y_test, verbose=0)[1]
  
droupout_accuracies_vals = list(droupout_accuracies.values())

plt.plot(0.1, droupout_accuracies_vals[0], "bo", label='0.1', color='black')
plt.plot(0.25, droupout_accuracies_vals[1], "bo", label='0.25', color='blue')
plt.plot(0.5, droupout_accuracies_vals[2],  "bo", label='0.5', color='red')
plt.plot(0.75, droupout_accuracies_vals[3], "bo", label='0.75', color="orange")
plt.plot(0.9, droupout_accuracies_vals[4], "bo", label='0.9', color="green")

plt.legend(loc="center right")
plt.xlabel("dropout rate")
plt.ylabel("accuracy")

plt.title("Dropout vs Accuracy")
plt.show()



# Part D

rate = 0.1
noise_accuracies = []

# best model from part c
model_d = keras.Sequential([
      keras.Input(shape=img_shape),
      layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dense(32, activation='relu'),
      layers.Dropout(rate),  
      layers.Dense(32, activation='relu'),
      layers.Dropout(rate),  
      layers.Dense(32, activation='relu'),
      layers.Dense(num_classes, activation="softmax"),
    ]) 
model_d.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model_d.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test,y_test))

for sigma in [0, 0.001, 0.01, 0.1, 0.25, 1]:
  # add noise to ALL images
  x_test_noise = []
  for ix,img in enumerate(x_test):
    x_test_noise.append(random_noise(img, mode='gaussian', mean=0, var=sigma*1))
  x_test_noise = np.array(x_test_noise)  
  
  noise_accuracies.append(model_d.evaluate(x_test_noise, y_test, verbose=0)[1])

plt.plot(0, noise_accuracies[0], "bo", label='control', color='black')
plt.plot(0.001, noise_accuracies[1], "bo", label='0.001', color='blue')
plt.plot(0.01, noise_accuracies[2],  "bo", label='0.01', color='red')
plt.plot(0.1, noise_accuracies[3], "bo", label='0.1', color="orange")
plt.plot(0.25, noise_accuracies[4], "bo", label='0.25', color="green")
plt.plot(1, noise_accuracies[5], "bo", label='1', color='purple')

plt.legend(loc="center right")

plt.title("Sigma (Noise) vs Accuracy - ALL")
plt.show()


rate = 0.1
noise_accuracies = []

# best model from part c
model_d = keras.Sequential([
      keras.Input(shape=img_shape),
      layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dense(32, activation='relu'),
      layers.Dropout(rate),  
      layers.Dense(32, activation='relu'),
      layers.Dropout(rate),  
      layers.Dense(32, activation='relu'),
      layers.Dense(num_classes, activation="softmax"),
    ]) 
model_d.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model_d.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test,y_test))

for sigma in [0, 0.001, 0.01, 0.1, 0.25, 1]:
  # add noise to 10% of images
  x_test_noise = []
  for ix,img in enumerate(x_test):
    random_num = random.randint(0,9)
    if random_num == 0:
      x_test_noise.append(random_noise(img, mode='gaussian', mean=0, var=sigma*1))
    else:
      x_test_noise.append(img)
  x_test_noise = np.array(x_test_noise)
  
  noise_accuracies.append(model_d.evaluate(x_test_noise, y_test, verbose=0)[1])

plt.plot(0, noise_accuracies[0], "bo", label='control', color='black')
plt.plot(0.001, noise_accuracies[1], "bo", label='0.001', color='blue')
plt.plot(0.01, noise_accuracies[2],  "bo", label='0.01', color='red')
plt.plot(0.1, noise_accuracies[3], "bo", label='0.1', color="orange")
plt.plot(0.25, noise_accuracies[4], "bo", label='0.25', color="green")
plt.plot(1, noise_accuracies[5], "bo", label='1', color='purple')

plt.legend(loc="center right")

plt.title("Sigma (Noise) vs Accuracy - 10%")
plt.show()



# Part E

rate = 0.1
rotated_accuracies = []

# best model from part c
model_e = keras.Sequential([
      keras.Input(shape=img_shape),
      layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dense(32, activation='relu'),
      layers.Dropout(rate),  
      layers.Dense(32, activation='relu'),
      layers.Dropout(rate),  
      layers.Dense(32, activation='relu'),
      layers.Dense(num_classes, activation="softmax"),
    ]) 
model_e.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model_e.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test,y_test))

for degrees in [0, 10, 20, 30, 45, 60]:
  # add noise to ALL images
  x_test_rotated = []
  for ix,img in enumerate(x_test):
    x_test_rotated.append(rotate(img, degrees))
  x_test_rotated = np.array(x_test_rotated)  
  rotated_accuracies.append(model_e.evaluate(x_test_rotated, y_test, verbose=0)[1])

plt.plot(0, rotated_accuracies[0], "bo", label='control', color='black')
plt.plot(10, rotated_accuracies[0], "bo", label='10', color='blue')
plt.plot(20, rotated_accuracies[1],  "bo", label='20', color='red')
plt.plot(30, rotated_accuracies[2], "bo", label='30', color="orange")
plt.plot(45, rotated_accuracies[3], "bo", label='45', color="green")
plt.plot(60, rotated_accuracies[4], "bo", label='60', color='purple')

plt.legend(loc="center left")

plt.title("Rotation Degree vs Accuracy")
plt.show()
