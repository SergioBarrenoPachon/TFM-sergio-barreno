import tensorflow as tf
import pydicom
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from imutils import paths
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, concatenate, Flatten, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.image import ssim
import numpy as np
import glob
from multiprocessing import Pool
import pandas as pd

# Dimensiones de las imágenes de entrada
img_width, img_height = 440,440

# Entrenar el modelo
epochs = 20
batch_size = 16

# Parámetro de la función de pérdida mixta RMSE + SSIM
alpha=0.3


# Carga de imagenes
def load_dicom_image(file_path):
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array
    return image

def load_dicom_images(file_paths):
    with Pool() as pool:
        images = pool.map(load_dicom_image, sorted(file_paths))
    images = np.array(images, dtype=np.float32)

    # Normalizar las imágenes en el rango [0, 1]
    max_value = np.max(images)
    images /= max_value
    # Asegurar que las imágenes tienen una dimensión de canal (escala de grises)
    images = np.expand_dims(images, axis=-1)
    return images

# Resto del código
image_files_noisy_1 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_1_20211201_164050/1-2 dose/*.IMA'))
image_files_noisy_2 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_2_20211201_164139/1-2 dose/*.IMA'))
image_files_noisy_3 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_3_20211201_164209/1-2 dose/*.IMA'))
image_files_noisy_4 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_4_20211201_164237/1-2 dose/*.IMA'))
image_files_noisy_5 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_5_20211201_164301/1-2 dose/*.IMA'))
image_files_noisy_6 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/02122021_1_20211202_160434/1-2 dose/*.IMA'))
image_files_noisy_7 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/02122021_2_20211202_160458/1-2 dose/*.IMA'))
image_files_noisy_8 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/02122021_3_20211202_160525/1-2 dose/*.IMA'))
image_files_noisy_9 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03012022_1_20220103_141016/1-2 dose/*.IMA'))
image_files_noisy_10 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03012022_2_20220103_141041/1-2 dose/*.IMA'))
image_files_noisy_11 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03012022_3_20220103_141110/1-2 dose/*.IMA'))
image_files_noisy_12 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03122021_1_20211203_165117/1-2 dose/*.IMA'))
image_files_noisy_13 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/03122021_2_20211203_165139/1-2 dose/*.IMA'))
image_files_noisy_14 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_1_20220104_143639/1-2 dose/*.IMA'))
image_files_noisy_15 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_2_20220104_143716/1-2 dose/*.IMA'))
image_files_noisy_16 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_3_20220104_143924/1-2 dose/*.IMA'))
image_files_noisy_17 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_4_20220104_143949/1-2 dose/*.IMA'))
image_files_noisy_18 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04112021_1_20211104_170047/1-2 dose/*.IMA'))
image_files_noisy_19 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05012022_1_20220105_151111/1-2 dose/*.IMA'))
image_files_noisy_20 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05012022_2_20220105_151149/1-2 dose/*.IMA'))
image_files_noisy_21 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_1_20211105_131722/1-2 dose/*.IMA'))
image_files_noisy_22 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_2_20211105_131821/1-2 dose/*.IMA'))
image_files_noisy_23 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_3_20211105_131907/1-2 dose/*.IMA'))
image_files_noisy_24 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_4_20211105_145054/1-2 dose/*.IMA'))
image_files_noisy_25 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/05112021_5_20211105_154512/1-2 dose/*.IMA'))
image_files_noisy_26 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06012022_1_20220106_160137/1-2 dose/*.IMA'))
image_files_noisy_27 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06012022_2_20220106_160212/1-2 dose/*.IMA'))
image_files_noisy_28 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06012022_3_20220106_161348/1-2 dose/*.IMA'))
image_files_noisy_29 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06122021_1_20211207_095014/1-2 dose/*.IMA'))
image_files_noisy_30 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06122021_2_20211207_095139/1-2 dose/*.IMA'))

train_image_files_noisy_l = image_files_noisy_1 + image_files_noisy_2 + image_files_noisy_3 + image_files_noisy_4 + image_files_noisy_5 + \
						image_files_noisy_6 + image_files_noisy_7 + image_files_noisy_8 + image_files_noisy_9 + image_files_noisy_10# + \
					#	image_files_noisy_11 + image_files_noisy_12 + image_files_noisy_13 + image_files_noisy_14 + image_files_noisy_15 + \
					#	image_files_noisy_16 + image_files_noisy_17 + image_files_noisy_18 + image_files_noisy_19 + image_files_noisy_20 + \
					#	image_files_noisy_21 + image_files_noisy_22 + image_files_noisy_23 + image_files_noisy_24
test_image_files_noisy_l = image_files_noisy_25 + image_files_noisy_26 + image_files_noisy_27# + \
					#	image_files_noisy_28 + image_files_noisy_29 + image_files_noisy_30



image_files_clean_1 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_1_20211201_164050/Full_dose/*.IMA'))
image_files_clean_2 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_2_20211201_164139/Full_dose/*.IMA'))
image_files_clean_3 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_3_20211201_164209/Full_dose/*.IMA'))
image_files_clean_4 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_4_20211201_164237/Full_dose/*.IMA'))
image_files_clean_5 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/01122021_5_20211201_164301/Full_dose/*.IMA'))
image_files_clean_6 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_1-6/02122021_1_20211202_160434/Full_dose/*.IMA'))
image_files_clean_7 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/02122021_2_20211202_160458/Full_dose/*.IMA'))
image_files_clean_8 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/02122021_3_20211202_160525/Full_dose/*.IMA'))
image_files_clean_9 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03012022_1_20220103_141016/Full_dose/*.IMA'))
image_files_clean_10 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03012022_2_20220103_141041/Full_dose/*.IMA'))
image_files_clean_11 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03012022_3_20220103_141110/Full_dose/*.IMA'))
image_files_clean_12 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_7-12/03122021_1_20211203_165117/Full_dose/*.IMA'))
image_files_clean_13 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/03122021_2_20211203_165139/Full_dose/*.IMA'))
image_files_clean_14 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_1_20220104_143639/Full_dose/*.IMA'))
image_files_clean_15 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_2_20220104_143716/Full_dose/*.IMA'))
image_files_clean_16 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_3_20220104_143924/Full_dose/*.IMA'))
image_files_clean_17 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04012022_4_20220104_143949/Full_dose/*.IMA'))
image_files_clean_18 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_13-18/04112021_1_20211104_170047/Full_dose/*.IMA'))
image_files_clean_19 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05012022_1_20220105_151111/Full_dose/*.IMA'))
image_files_clean_20 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05012022_2_20220105_151149/Full_dose/*.IMA'))
image_files_clean_21 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_1_20211105_131722/Full_dose/*.IMA'))
image_files_clean_22 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_2_20211105_131821/Full_dose/*.IMA'))
image_files_clean_23 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_3_20211105_131907/Full_dose/*.IMA'))
image_files_clean_24 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_19-24/05112021_4_20211105_145054/Full_dose/*.IMA'))
image_files_clean_25 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/05112021_5_20211105_154512/Full_dose/*.IMA'))
image_files_clean_26 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06012022_1_20220106_160137/Full_dose/*.IMA'))
image_files_clean_27 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06012022_2_20220106_160212/Full_dose/*.IMA'))
image_files_clean_28 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06012022_3_20220106_161348/Full_dose/*.IMA'))
image_files_clean_29 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06122021_1_20211207_095014/Full_dose/*.IMA'))
image_files_clean_30 = sorted(glob.glob('/LUSTRE/users/sbarreno/proyectos/TFM/data/Subject_25-30/06122021_2_20211207_095139/Full_dose/*.IMA'))


train_image_files_clean_l = image_files_clean_1 + image_files_clean_2 + image_files_clean_3 + image_files_clean_4 + image_files_clean_5 + \
						image_files_clean_6 + image_files_clean_7 + image_files_clean_8 + image_files_clean_9 + image_files_clean_10# + \
				        #	image_files_clean_11 + image_files_clean_12 + image_files_clean_13 + image_files_clean_14 + image_files_clean_15 + \
					#	image_files_clean_16 + image_files_clean_17 + image_files_clean_18 + image_files_clean_19 + image_files_clean_20 + \
					#	image_files_clean_21 + image_files_clean_22 + image_files_clean_23 + image_files_clean_24
test_image_files_clean_l = image_files_clean_25 + image_files_clean_26 + image_files_clean_27# + \
					#	image_files_clean_28 + image_files_clean_29 + image_files_clean_30

train_image_files_noisy = load_dicom_images(train_image_files_noisy_l)
train_image_files_clean = load_dicom_images(train_image_files_clean_l)
test_image_files_noisy = load_dicom_images(test_image_files_noisy_l)
test_image_files_clean = load_dicom_images(test_image_files_clean_l)
print('loaded images')


# Definir la función de pérdida mixta RMSE + SSIM
def mixed_loss(y_true, y_pred, alpha=1):

    rmse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))

    ssim1 = 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

    return alpha*rmse + (1-alpha)*ssim1


# Crear discriminador
def crear_discriminador(input_shape):
    img_input = Input(shape=input_shape)

    x = Conv2D(16, kernel_size=3, padding='same')(img_input)
    x = Conv2D(32, kernel_size=3, padding='same')(x)

    x = Flatten()(x)

    x = Dense(10, activation='relu')(x)  # Capa completamente conectada

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=img_input, outputs=x)
    # model.compile(optimizer=d_optimizer, loss='binary_crossentropy')

    return model

discriminador = crear_discriminador(input_shape=(img_width, img_height, 2))


# Definir una arquitectura U-Net
def unet(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))

    # Encoder
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bridge
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))

    # Decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))

    # Output
    model.add(Conv2D(1, 1, activation='relu'))

    return model
unet_model = unet(input_shape=(img_width, img_height, 3))


# Train function
# Optimizadores.
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)

# Funcion de perdida.
loss_fn = keras.losses.BinaryCrossentropy() 

@tf.function
def train_step(noisy_images,real_images,noisy_images_25D):

    combined_images_gen = tf.concat([noisy_images, unet_model(noisy_images_25D)], axis=-1)
    combined_images_real = tf.concat([noisy_images, real_images], axis=-1)
    print("Dimensiones de combined_images_gen:", combined_images_gen.shape)
    print("Dimensiones de combined_images_real:", combined_images_real.shape)

    labels_gen = tf.zeros((noisy_images.shape[0], 1))
    labels_real = tf.ones((real_images.shape[0], 1))

    # Entrenar el discriminador
    with tf.GradientTape() as tape:
        d_loss = loss_fn(labels_gen, discriminador(combined_images_gen)) + loss_fn(labels_real, discriminador(combined_images_real))

    grads = tape.gradient(d_loss, discriminador.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminador.trainable_weights))

    # Como se supone que deberían dar las imagenes generadas para engañar
    misleading_labels = tf.ones((noisy_images.shape[0], 1))

    # Entrenar el generador
    with tf.GradientTape() as tape:
        predictions = discriminador(tf.concat([noisy_images, unet_model(noisy_images_25D)], axis=-1))
        g_loss = loss_fn(misleading_labels, predictions)

    grads = tape.gradient(g_loss, unet_model.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, unet_model.trainable_weights))

    return d_loss, g_loss


# Ejecutar entrenamiento
print("inicio entrenamiento")
input_25D_train = []
input_25D_test = []
real_image_train = train_image_files_clean
real_image_test = test_image_files_clean

result_train = []
result_test = []
input_image_train = train_image_files_noisy
input_image_test = test_image_files_noisy	
print('formato de input_25D_train es ', input_image_train.shape)
print("end input_25D_load")

for image in range(len(input_image_train)):
    anterior = image - 1
    actual = image
    posterior = image + 1
    if image == 0:
        anterior = image
    if image == len(input_image_train) - 1:
        posterior = image
    images_to_app = np.concatenate([input_image_train[anterior],input_image_train[actual],input_image_train[posterior]], axis = -1)
    input_25D_train.append(images_to_app)
input_25D_train = np.stack(input_25D_train, axis = 0)

for image in range(len(input_image_test)):
    anterior = image - 1
    actual = image
    posterior = image +1
    if image == 0:
        anterior = image
    if image == len(input_image_test) - 1:
        posterior = image
    images_to_app = np.concatenate([input_image_test[anterior],input_image_test[actual],input_image_test[posterior]], axis = -1)
    input_25D_test.append(images_to_app)
input_25D_test = np.stack(input_25D_test, axis = 0)

start_time = time.time()

for epoch in range(epochs):
    print("\nStart epoch", epoch)

    num_batches = len(train_image_files_noisy) // batch_size
    for batch in range(num_batches):
        batch_noisy = train_image_files_noisy[batch * batch_size : (batch + 1) * batch_size]
        batch_clean = train_image_files_clean[batch * batch_size : (batch + 1) * batch_size]
        batch_noisy_25D = input_25D_train[batch * batch_size : (batch + 1) * batch_size]

        # Entrenamiento por batches.
        d_loss, g_loss = train_step(batch_noisy,batch_clean,batch_noisy_25D)
        # Logging.
        if batch % 15 == 0:
            elapsed_time = time.time() - start_time
            print(elapsed_time)
            print("discriminator loss at epoch %d and step %d: %.2f" % (epoch, batch, d_loss))
            print("adversarial loss at epoch %d and step %d: %.2f" % (epoch, batch, g_loss))

    result_train_step = []
    result_test_step = []
    for i in range(input_25D_train.shape[0]):
        output_image_train = unet_model.predict(np.expand_dims(input_25D_train[i], axis=0))
        result_train_step.append(mixed_loss(real_image_train[i],output_image_train))    
    result_train.append(tf.reduce_mean(result_train_step))
    for i in range(input_25D_test.shape[0]):
        output_image_test = unet_model.predict(np.expand_dims(input_25D_test[i], axis=0))
        result_test_step.append(mixed_loss(real_image_test[i],output_image_test))
    result_test.append(tf.reduce_mean(result_test_step))

# Crea un DataFrame de pandas con los vectores
result = {'Train': result_train, 'Test': result_test}
result = pd.DataFrame(result)

# Nombre del archivo de Excel de salida
csv_filename = 'errores_GAN_25D_09.csv'

# Escribe el DataFrame en un archivo Excel
result.to_csv(csv_filename, sep=';')

print(f'Se ha creado el archivo "{csv_filename}" con éxito.')


