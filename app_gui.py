import tkinter as tk
from tkinter import *
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage
import os
from os import listdir
from os.path import isdir, join
from face_extractor import data_collection
from face_training import train_model
from recognize import main_app

names = set()
train_path = r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\train/'
valid_path = r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\validation/'
info_path = r'C:\Users\alanj\Documents\Python\Practice\face_recognition\profile_data/'

class MainUI(tk.Tk):
    
    
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        names = [name for name in listdir(train_path) if isdir(join(train_path, name))]
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight='bold')        
        self.title('Welcome to Intelligent Face Recognition World')
        self.resizable(False, False)
        self.geometry("650x250")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        '''for F in (StartPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")'''
        page_name = StartPage.__name__
        frame = StartPage(parent=container, controller=self)
        self.frames[page_name] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")
        
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()
            
class StartPage(tk.Frame):
    
    def __init__(self, parent, controller):
        
        def make_new_profile():
            name = t1.get()
            nTrainPath = join(train_path, name)
            nValidPath = join(valid_path, name)
            nInfoPath = join(info_path, name)
            if not os.path.exists(nTrainPath):
                os.mkdir(nTrainPath)
                tk.messagebox.showinfo(title="Info", message="Folder created")
            if not os.path.exists(nValidPath):
                os.mkdir(nValidPath)
            if not os.path.exists(nInfoPath):
                os.mkdir(nInfoPath)
            age = t2.get()
            phone = t3.get()
            data_collection(name, age, phone)
            messagebox.Message("Profile registered. Please train the model")
            
        def train():
            train_model()
            
        def recognizer():
            main_app()
            
        tk.Frame.__init__(self, parent)
        self.controller = controller
        l1=tk.Label(self,text="Name",font=("Helvetica", 16))
        l1.grid(column=0, row=0, padx=(215, 0), pady=(20, 0))
        t1=tk.Entry(parent,width=20,bd=5)
        t1.grid(column=1, row=0, pady=(20, 0))
        
        l2=tk.Label(parent,text="Age",font=("Helvetica", 16))
        l2.grid(column=0, row=1, padx=(225, 0))
        t2=tk.Entry(parent,width=20,bd=5)
        t2.grid(column=1, row=1)
        
        l3=tk.Label(parent,text="Phone No.",font=("Helvetica", 16))
        l3.grid(column=0, row=2, padx=(175, 0))
        t3=tk.Entry(parent,width=20,bd=5)
        t3.grid(column=1, row=2)
        
        b1=tk.Button(parent,text="Training",font=("Helvetica",16),
                     bg='orange',fg='black', command=train)  # add command=function for training
        b2=tk.Button(parent,text="Register",font=("Helvetica",16),
                     bg='pink',fg='black', command=make_new_profile)
        b3=tk.Button(parent,text="Recognize",font=("Helvetica",16),
                     bg='cyan',fg='black', command=recognizer)
        b2.grid(column=2, row=4, padx=(100, 0), pady=(10, 0))
        b3.grid(column=1, row=4, pady=(10, 0))
        b1.grid(column=0, row=4, pady=(10, 0))
        
app = MainUI()
app.mainloop()