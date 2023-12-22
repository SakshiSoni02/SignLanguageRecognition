#import glob
#from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def text ():
 global entry
 string = entry.get()
 image_paths = []
 strin = string
 img = cv2.imread("text/aA.jpg")
 for i, v in enumerate(strin):
    if v == 'a':
        image_paths.append('text/aA.jpg')
    elif v == 'b':
        image_paths.append('text/bB.jpg')
    elif v == 'c':
        image_paths.append('text/cC.jpg')
    elif v == 'd':
        image_paths.append('text/dD.jpg')
    elif v == 'e':
        image_paths.append('text/eE.jpg')
    elif v == 'f':
        image_paths.append('text/fF.jpg')
    elif v == 'g':
        image_paths.append('text/gG.jpg')
    elif v == 'h':
        image_paths.append('text/hH.jpg')
    elif v == 'i':
        image_paths.append('text/iI.jpg')
    elif v == 'j':
        image_paths.append('text/jJ.jpg')
    elif v == 'k':
        image_paths.append('text/kK.jpg')
    elif v == 'l':
        image_paths.append('text/lL.jpg')
    elif v == 'm':
        image_paths.append('text/mM.jpg')
    elif v == 'n':
        image_paths.append('text/nN.jpg')
    elif v == 'o':
        image_paths.append('text/oO.jpg')
    elif v == 'p':
        image_paths.append('text/pP.jpg')
    elif v == 'q':
        image_paths.append('text/qQ.jpg')
    elif v == 'r':
        image_paths.append('text/rR.jpg')
    elif v == 's':
        image_paths.append('text/sS.jpg')
    elif v == 't':
        image_paths.append('text/tT.jpg')
    elif v == 'u':
        image_paths.append('text/uU.jpg')
    elif v == 'v':
        image_paths.append('text/vV.jpg')
    elif v == 'w':
        image_paths.append('text/wW.jpg')
    elif v == 'x':
        image_paths.append('text/xX.jpg')
    elif v == 'y':
        image_paths.append('text/yY.jpg')
    elif v == 'z':
        image_paths.append('text/zZ.jpg')
    elif v == ' ':
        image_paths.append('text/blank.jpg')
    # Create subplots to display the images
 fig, axs = plt.subplots(1, len(image_paths))

    # Iterate through the images and display them
 for i, image_path in enumerate(image_paths):
        # Read the image from the file
    image = plt.imread(image_path)

        # Display the image on the corresponding subplot
    axs[i].imshow(image)
    axs[i].axis('off')

    # Show the figure with the images
 plt.show()

   # Show the figure with the images
 plt.show()
   # cv2.imshow('HORIZONTAL' , img )

#Import the required Libraries
from tkinter import *
from tkinter import ttk

#Create an instance of Tkinter frame
win= Tk()

#Set the geometry of Tkinter frame
win.geometry("750x250")

win.title ('Text to Sign ')



#Initialize a Label to display the User Input
label=Label(win, text="Enter the text you want to Translate", font=("Courier 22 bold"))
label.pack()

#Create an Entry widget to accept User Input
entry= Entry(win, width= 40)
entry.focus_set()
entry.pack()

#Create a Button to validate Entry Widget
ttk.Button(win, text= "Convert ",width= 20, command=text).pack(pady=20)

win.mainloop()



