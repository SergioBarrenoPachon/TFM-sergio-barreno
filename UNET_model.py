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
import gc

# Dimensiones de las imágenes de entrada
img_width, img_height = 440,440
# Entrenar el modelo
epochs = 30 
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


# Definir una arquitectura U-Net más sencilla
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

unet_model = unet(input_shape=(img_width, img_height, 1))



# Instantiate one optimizer for the discriminator and another for the generator.
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)


# Ejecutar entrenamiento
input_image_train = train_image_files_noisy
real_image_train = train_image_files_clean
input_image_test = test_image_files_noisy
real_image_test = test_image_files_clean


unet_model.compile(optimizer='adam', loss='mean_squared_error')
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
history=unet_model.fit(input_image_train,real_image_train,epochs=epochs,batch_size=batch_size,validation_split=0.2, callbacks=[checkpoint]) 
unet_model.save("best_model.h5")

result_train=history.history['loss']
result_test=history.history['val_loss']          

# Crea un DataFrame de pandas con los vectores
result = pd.DataFrame(history.history)

# Nombre del archivo de Excel de salida
csv_filename = 'train_UNET_02_00.csv'

# Escribe el DataFrame en un archivo Excel
result.to_csv(csv_filename, index=False)

print(f'Se ha creado el archivo "{csv_filename}" con éxito.')

best_model = load_model("best_model.h5")

result_test_step1 = []
result_test_step2 = []
for i in range(input_image_test.shape[0]):
    output_image_test = unet_model.predict(np.expand_dims(input_image_test[i], axis = 0))
    result_test_step1.append(tf.reduce_mean(tf.keras.losses.mean_squared_error(real_image_test[i],output_image_test)))
    result_test_step2.append( 1 - tf.reduce_mean(ssim(real_image_test[i],output_image_test, max_val=1.0)))
result_test1=tf.reduce_mean(result_test_step1).numpy()
result_test2=tf.reduce_mean(result_test_step2).numpy()
csv_filename_2 = 'error_UNET_02_00.csv'
result_test = pd.DataFrame({'ECM':np.array([result_test1]),'SSIM':np.array([result_test2])})
result_test.to_csv(csv_filename_2, index=False)

generated_image = best_model.predict(np.expand_dims(input_image_test[234], axis = 0))

plt.figure(figsize=(12, 4))

# Imagen ruidosa
plt.subplot(1, 3, 1)
plt.imshow(input_image_test[234], cmap='hot_r')
plt.title('Imagen Ruidosa')
plt.axis('off')

# Imagen generada
plt.subplot(1, 3, 2)
plt.imshow(generated_image.squeeze(), cmap='hot_r')  # Utiliza .squeeze() para eliminar dimensiones adicionales
plt.title('Imagen Generada')
plt.axis('off')

# Imagen limpia real
plt.subplot(1, 3, 3)
plt.imshow(real_image_test[234], cmap='hot_r')
plt.title('Imagen Limpia Real')
plt.axis('off')

plt.show()

png_filename = "image_UNET_02_00_A.png"
plt.savefig(png_filename, bbox_inches='tight')

print(f'Se ha creado el archivo "{png_filename}" con éxito.')

# SEGUNDA IMAGEN

generated_image = best_model.predict(np.expand_dims(input_image_test[270], axis = 0))

plt.figure(figsize=(12, 4))

# Imagen ruidosa
plt.subplot(1, 3, 1)
plt.imshow(input_image_test[270], cmap='hot_r')
plt.title('Imagen Ruidosa')
plt.axis('off')

# Imagen generada
plt.subplot(1, 3, 2)
plt.imshow(generated_image.squeeze(), cmap='hot_r')  # Utiliza .squeeze() para eliminar dimensiones adicionales
plt.title('Imagen Generada')
plt.axis('off')

# Imagen limpia real
plt.subplot(1, 3, 3)
plt.imshow(real_image_test[270], cmap='hot_r')
plt.title('Imagen Limpia Real')
plt.axis('off')

plt.show()

png_filename = "image_UNET_02_00_B.png"
plt.savefig(png_filename, bbox_inches='tight')

print(f'Se ha creado el archivo "{png_filename}" con éxito.')


print('END PYTHON')
