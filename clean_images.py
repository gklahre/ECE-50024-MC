import tkinter as Tkinter
import numpy as np
from PIL import Image, ImageTk
from sys import argv
import csv
import cv2
import os
import math

old_file = open("train.csv")
current_line = old_file.readline()
current_line = old_file.readline()
gui_x = [100, 325, 550, 775, 1000]
gui_y = [100, 100, 100, 100, 100]
print(current_line)

if os.path.exists("mytrain.csv") is False:
    new_file = open("mytrain.csv", "w")
else:
    new_file = open("mytrain.csv", "a")

#window = Tkinter.Tk(className="bla")

rlist = current_line.split(',')
print(rlist)

r = None
def callback(event):
    global r 
    if r is not None:
        canvas.delete(r)
    r = canvas.create_rectangle(event.x,event.y, event.x+200, event.y+200, outline='black')

def save():
    #TODO: Make sure it's getting save under the right name
    pass
    image.crop(r.x, r.y, x+200, y+200)
    image.save("saved/" + fn)
    new_file.write(current_line)

def discard():
    pass
    os.remove(fn)

def open_new():
    global current_line
    global image
    global image_tk
    global temp_i
    global canvas
    current_line = old_file.readline()
    rlist = current_line.split(',')
    image = Image.open("mytrain/" + rlist[1])
    f_list = faces(image)
    print("Image #: " + rlist[0])
    
    temp_i = []
    if f_list is None:
        open_new()
        return
    #elif f_list.len > 20:
    #   get_new()
    #else: 
    print(len(f_list))
    for i in range(min(len(f_list), 5)):
        temp_i.append(ImageTk.PhotoImage(f_list[i]))
        canvas.create_image(gui_x[i], gui_y[i], image=temp_i[i], tags=("image"))
    #image_tk = ImageTk.PhotoImage(image)
    #i = canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk, tags=("image"))
    j = canvas.create_text(250,800, text=current_line, fill="black", font=("Helvetica 15 bold"), tags=('info'))

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

def decision(event):
    #print(event.char)
    if event.char == "1":
        #save()
        canvas.delete('image')
        canvas.delete('info')
        open_new()

    elif event.char == "0":
        #discard()
        canvas.delete('all')
        canvas.delete('info')
        open_new()
      
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

image = Image.open("mytrain/" + rlist[1])
#canvas = Tkinter.Canvas(window, width=1200, height=1000)
#canvas.pack()
#image_tk = ImageTk.PhotoImage(image)
#f_list = faces(image)
#if f_list.len == 0:
#   get_new()
#elif f_list.len > 20:
#   get_new()
#else: 
#temp_i = []
#for i in range(len(f_list)):
#    temp_i.append(ImageTk.PhotoImage(f_list[i]))
#    canvas.create_image(gui_x[i], gui_y[i], image=temp_i[i], tags=("image"))

#i = canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk, tags=("image"))
#j = canvas.create_text(250,800, text=current_line, fill="black", font=("Helvetica 15 bold"), tags=("info"))

#canvas.bind("<Button-1>", callback)
#window.bind("<Key>", decision)
#Tkinter.mainloop()


# Category Dict
g = open("category.csv")
label_dict = dict()
inverse_dict = dict()
lines = g.readlines() 
for line in lines:
    line = line.split(',')
    dir = 'D:/Documents/Homework/ECE 50024/full/' + line[1].rstrip()
    if not os.path.exists(dir):
        os.makedirs(dir)
    #label_dict[line[1]]= line[0]
    #inverse_dict[line[0]] = line[1]
g.close()

f = open("mytrain.csv")
lines = f.readlines()
l = 0
for line in lines:
    l += 1
    rlist = line.split(',')
    if(l % 100 == 0):
        print(l)
    new_dir = 'full/' + rlist[2].rstrip() + "/" + rlist[1]
    os.rename('mytrain/' + rlist[1], 'full/' + r)


            
            


