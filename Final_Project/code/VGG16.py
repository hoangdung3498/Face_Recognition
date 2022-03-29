import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Input, Lambda ,Dense ,Flatten , Dropout , GlobalAveragePooling2D

#vgg 16 model
classifier_vgg16 = VGG16(input_shape= (224,224,3),include_top=False,weights='imagenet')
classifier_vgg16.summary()

#not train top layers
for layer in classifier_vgg16.layers:
    layer.trainable = False

#adding extra layers for our class/images
main_model = classifier_vgg16.output
main_model = GlobalAveragePooling2D()(main_model)
main_model = Dense(512,activation='relu')(main_model)
main_model = Dropout(0.4)(main_model)
main_model = Dense(512,activation='relu')(main_model)
main_model = Dropout(0.4)(main_model)
main_model = Dense(512,activation='relu')(main_model)
main_model = Dropout(0.4)(main_model)
main_model = Dense(3,activation='softmax')(main_model)

#compiling
model = Model(inputs = classifier_vgg16.input , outputs = main_model)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#use the image data generator to import the images from the dataset
#data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

#makes sure you provide the same target as initialised for the image size
training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(224, 224),
                                                 batch_size=1,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/val',
                                            target_size=(224, 224),
                                            batch_size=1,
                                            class_mode='categorical',
                                            shuffle=False)

#fit the model
#it will take some time to train
nb_train_samples=75
batch_size=1
nb_validation_samples=5
history = model.fit_generator(training_set,
                              validation_data=test_set,
                              epochs=10,
                              steps_per_epoch=nb_train_samples // batch_size,
                              validation_steps=nb_validation_samples // batch_size)
model.save('model.h5')



