from random import random
from tkinter import *
from tkinter import ttk, font, filedialog, Entry

from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time
import keras
from inference import infe
from tensorflow.keras import backend as K


import cv2
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)



#Inicio de la aplicacion


class Backend:

    def save_results():
        save = showinfo(title="Guardar", message="Los datos se guardaron con éxito.")
        return save

    def create():
        
        create= showinfo(title="PDF", message="El PDF fue generado con éxito.")
        return create

    def delet():
        dele = showinfo(title="Borrar", message="Los datos se borraron con éxito")
        return dele
    
    def con_delet():
        confirmar = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        return confirmar
