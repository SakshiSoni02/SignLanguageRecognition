import tkinter
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt
import os



m = tkinter.Tk()
m.geometry('850x450')
m.title("Sign Language")
m.configure(bg="light blue")

m = tkinter.Label(text="Sign Language Detection",bg="light blue")
m.config(font=('Helvetica bold',20))
m.place(x=270,y=20)

np.set_printoptions(suppress=True)
model_number = load_model("keras_Model.h5", compile=False)
class_names_number = open("labels.txt", "r").readlines()

model_word = load_model("keras_Model.h5", compile=False)
class_names_word = open("labels.txt", "r").readlines()

cap = cv2.VideoCapture(0)

def num():
    m.destroy()
    os.system('text.py')

def word():
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            rimage = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            imagee = np.asarray(rimage, dtype=np.float32).reshape(1, 224, 224, 3)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            imagee = (imagee / 127.5) - 1

            # Predicts the model
            prediction = model_word.predict(imagee)
            index = np.argmax(prediction)
            class_name = class_names_word[index]
            confidence_score = prediction[0][index]

            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

            a = class_name[2:-1]+" "+str(np.round(confidence_score * 100))[:-2]+ "%"

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(image,a, (0, 100), font, 2, (0, 0, 255), 3)  # text,coordinate,font,size of text,color,thickness of font

            f = open("word.txt", "r")
            n= f.read()
            cv2.putText(image,n, (0, 300), font, 1, (0, 255, 0), 3)  # text,coordinate,font,size of text,color,thickness of font


            if cv2.waitKey(1) == ord('a'):
                f = open("word.txt", "a")
                f.write(class_name[2:-1])
                f.close()

            if cv2.waitKey(2) == ord('s'):
                f = open("word.txt", "a")
                f.write(" ")
                f.close()

            if cv2.waitKey(3) == ord('c'):
                f = open("word.txt", "w")
                f.write(" ")
                f.close()


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            #cv2.imshow('Number Detection', cv2.flip(image, 1))
            cv2.imshow('Number Detection', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()



def number():
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            rimage = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            imagee = np.asarray(rimage, dtype=np.float32).reshape(1, 224, 224, 3)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            imagee = (imagee / 127.5) - 1

            # Predicts the model
            prediction = model_number.predict(imagee)
            index = np.argmax(prediction)
            class_name = class_names_number[index]
            confidence_score = prediction[0][index]

            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

            a = class_name[2:]+" "+str(np.round(confidence_score * 100))[:-2]+ "%"

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(image,a, (0, 100), font, 2, (0, 0, 255), 3)  # text,coordinate,font,size of text,color,thickness of font

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            #cv2.imshow('Number Detection', cv2.flip(image, 1))
            cv2.imshow('Number Detection', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()

btn = tkinter.Button(text='Text to Image', width=25, height=2, bg="yellow", fg="black", font=('times new roman',15,'italic'),command=num)
btn.place(x=300, y=100)
#
# btn1 = tkinter.Button(text='Character Detection', width=25, height=2, bg="yellow", fg="black", font=('times new roman',15,'italic'))
# btn1.place(x=450, y=100)

btn2 = tkinter.Button(text='Open Webcam', width=25, height=2, bg="yellow", fg="black", font=('times new roman',15,'italic'),command=word)
btn2.place(x=300, y=200)

m.mainloop()