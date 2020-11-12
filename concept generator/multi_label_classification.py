from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import xlsxwriter,xlrd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#进行配置，每个GPU使用60%上限现存
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())
#数据集
ExcelFile=xlrd.open_workbook('tag_dataset/new/tag_datalabel_identity_v4.xlsx')
sheet=ExcelFile.sheet_by_index(0)
# initialize hyperparameters - the number of epochs, initial learning rate, batch size, and image dimensions

EPOCHS = 60
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)


print("[INFO] Loading images...")
imagePaths=[]

for i in range(0, sheet.nrows):
    path = sheet.cell(i, 0).value.encode('utf-8').decode('utf-8-sig')
    imagePaths.append(path)


# initialize the data and labels
data = []

post_path=imagePaths[0]
# loop over the input images
for imagePath in imagePaths:
    
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    except:
        image = cv2.imread(post_path)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

labels=np.load('/media/gantian/New Volume/Experiment/Influencer_2020/tag_dataset/tag_dataset/new/tag_datalabel_identity.npy')
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# increase the training set size using data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
lens=len(labels[0])
print(lens)
# initialize the model and use sigmoid activation function in the final layer of the network
print("[INFO] Compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=lens,
                            final_act="sigmoid")

# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

# train the network
print("[INFO] Training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=2)

# save the model to disk
print("[INFO] Serializing network...")
model.save(args["model"])

# plot the training loss
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training/Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("identity_loss.png")

# plot the training accuracy
plt.clf()
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["binary_accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_binary_accuracy"], label="val_acc")
plt.title("Training/Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("identity_accuracy.png")
