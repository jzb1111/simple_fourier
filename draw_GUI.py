# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 23:31:07 2023

@author: asus
"""

import tkinter as tk
import utils
import PIL.ImageGrab as ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Draw():
    def __init__(self):
 
        # Defining title and Size of the Tkinter Window GUI
        
        self.pointer = "black"
        self.background=None
        self.eraser_btn=None
        self.pen_btn=None
        self.pre_btn=None
        self.root=None
        self.GUI()
        
    def GUI(self):
        self.root=tk.Tk()
        self.root.title("画图工具Python")
        self.root.geometry("500x400")
        
        self.eraser_btn = tk.Button(self.root, text="Eraser", command=self.eraser, width=15)
        self.eraser_btn.place(x=10, y=50)
        
        self.pen_btn = tk.Button(self.root, text="Pen", command=self.pen, width=15)
        self.pen_btn.place(x=10, y=100)
        
        self.pre_btn=tk.Button(self.root, text="gen_fourier_anim", command=self.gen_fourier_anim, width=15)
        self.pre_btn.place(x=10, y=150)
        
        self.background = tk.Canvas(self.root, bg='white', bd=5, relief=tk.GROOVE, height=300, width=300)
        self.background.place(x=140, y=50)
        self.background.bind("<B1-Motion>", self.paint)
        self.root.mainloop()
    
    def paint(self,event):
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer,width=13)
        
    def eraser(self):
        self.pointer = 'white'
    
    def pen(self):
        self.pointer = 'black'
    
    def gen_fourier_anim(self):
        x = self.root.winfo_rootx() + self.background.winfo_x()
        y = self.root.winfo_rooty() + self.background.winfo_y()
        x1 = x + self.background.winfo_width()
        y1 = y + self.background.winfo_height()
        tmp=ImageGrab.grab().crop((x, y, x1, y1))
        #print(tmp.load()[50,50])
        rgb=tmp.load()
        bin_im=np.zeros((self.background.winfo_height(),self.background.winfo_width()))
        for i in range(self.background.winfo_height()):
            for j in range(self.background.winfo_width()):
                bin_im[i][j]=1 if sum(rgb[j,i])>255 else 0
        bin_im=bin_im.astype(np.uint8)
        contours,hier=cv2.findContours(bin_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        fslis=utils.contours2fourier(contours)
        #dflis=utils.fourier2contours(fslis)
        anim=utils.play_anim(fslis,(len(bin_im),len(bin_im[0])))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter('draw_anim.avi', fourcc, fps=50, frameSize=(len(bin_im[0]),len(bin_im)))
        for i in range(len(anim)):
            cv2.imshow('circle',anim[i])
            cv2.waitKey(10)
            #for j in range(20):
            anim_i=anim[i].astype(np.uint8)
            
            anim_i = np.stack([anim_i,anim_i,anim_i],axis=-1)*255#cv2.cvtColor(anim_i, cv2.COLOR_GRAY2BGR)*255
            #print(anim_i[150,150])
            out_video.write(anim_i)
        out_video.release()
        cv2.destroyAllWindows()
        print('out ok')
Draw()