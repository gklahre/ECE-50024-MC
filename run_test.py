import tensorflow as tf
import numpy as np
from PIL import Image
import math
import cv2
import pathlib
import glob

def set_x(s: int, x, w) -> int:
    global x1
    global x2
    image_x = image.size[0]
    t = math.ceil((s-w) / 2)
    b= math.floor((s-w)/2)
    x1 = x - t
    x2 = x + w + b
    if (x1 < 0):
        d = -x1
        x2 = x2 + d
        x1 = 0
        if(x2 > image_x):
            d = x2 - image_x
            x2 = image_x
            return s-d
    if (x2 > image_x):
        d = x2 - image_x
        x1 = x1 - d
        x2 = image_x
        if(x1 < 0):
            d = -x1
            x1 = 0
            return s-d
    return 0

def set_y(s, y, h) -> int:
    global y1
    global y2
    image_y = image.size[1]
    t = math.ceil((s-h) / 2)
    b= math.floor((s-h)/2)
    y1 = y - t
    y2 = y + h + b
    if (y1 < 0):
        d = -y1
        y2 = y2 + d
        y1 = 0
        if(y2 > image_y):
            d = y2 - image_y
            y2 = image_y
            return s-d
    if (y2 > image_y):
        d = y2 - image_y
        y1 = y1 - d
        y2 = image_y
        if(y1 < 0):
            d = -y1
            y1 = 0
            return s-d
    return 0


def faces(image_i) -> list:
    pass
    #TODO: list faces, and crop them appropriately to 200x200
    npi = np.asarray(image_i)
    try:
        image = cv2.cvtColor(npi, cv2.COLOR_BGR2GRAY)
    except:
        image = npi
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor= 1.01, minSize=(100,100), minNeighbors=6)
    pfaces = profile_cascade.detectMultiScale(image,scaleFactor=1.01, minSize = (100,100), minNeighbors=6)
    r_list = []
    for (x, y, w, h) in faces:
        if w > h:
            s = math.ceil(w * 1.25)
        else :
            s = math.ceil(h * 1.25)
        if s % 2 == 1:
            s += 1
        c = set_x(s, x, w)
        if c != 0:
            s = c
        c = set_y(s, y, h)
        if c != 0:
            set_x(c, x, w)
        t_image = image_i.crop((x1,y1,x2,y2))
        t_image.thumbnail((200, 200))
        r_list.append(t_image)

    for (x, y, w, h) in pfaces:
        if w > h:
            s = math.ceil(w * 1.25)
        else :
            s = math.ceil(h * 1.25)
        if s % 2 == 1:
            s += 1
        c = set_x(s, x, w)
        if c != 0:
            s = c
        c = set_y(s, y, h)
        if c != 0:
            set_x(c, x, w)
        t_image = image_i.crop((x1,y1,x2,y2))
        t_image.thumbnail((200, 200))
        r_list.append(t_image)

        return r_list
    

model = tf.keras.models.load_model('model_2_2.keras')

model.summary()

my_data = pathlib.Path("mytrain").with_suffix('')
batch_size = 32
img_height = 200
img_width = 200

training_data = tf.keras.utils.image_dataset_from_directory(my_data,validation_split=0.2,subset="training",seed=200,image_size=(img_height, img_width), batch_size=batch_size)
class_names = training_data.class_names

test = []
list = glob.glob("test/*.jpg")
correctness_guess = 0
total = 0
file = open("submit.csv", "w+")
file.write("Id,Category\n")
for img in list:
    number = img.split(".")[0].split("\\")[1]
    select = -1
    highest = -1
    img = Image.open(img).convert("RGB")
    global image 
    image = img
    face = faces(img)

    if face is not None:
        for f in face:
            #f = np.asarray(f).astype('float32')
            #fa = tf.io.decode_image(f,channels=3)
            fa = tf.image.resize(f, (200,200))
            #fa = tf.keras.preprocessing.image.smart_resize(fa,(200,200))
            fa = tf.expand_dims(fa, 0)
            pred = model.predict(fa)
            score = tf.nn.softmax(pred)
            s = np.argmax(score)
            h = np.max(score)
            if h > highest:
                highest = h
                select = s
    #img = np.asarray(img).astype(str)
    #img = tf.cast(img,dtype=str)     
    #imga = tf.io.decode_image(img,channels=3)
    imga = tf.image.resize(img, (200,200))
    imga = tf.expand_dims(imga, 0)
    pred = model.predict(imga)
    score = tf.nn.softmax(pred[0])
    s = np.argmax(score)
    h = np.max(score)
    if h > highest:
        highest = h
        select = s
    #print("This image most likely belongs to {} with a {:.2f} percent confidence."
    #.format(class_names[select], 100 * highest)
    file.write(number + "," + class_names[select] + "\n")
    
    correctness_guess += highest
    total += 1
    if(total % 100 == 0):
        print(total)
        print(correctness_guess/total)

print(correctness_guess)
print(total)
print(correctness_guess/total)
