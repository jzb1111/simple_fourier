# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 23:32:16 2023

@author: asus
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def fourier_fit(lis,f_n):
    #lis:曲线的点集lis={p0,p1,p2...},p0=[x0,y0],...
    #f_n:级数的长度
    f_num=[0]
    f_num+=[i+1 for i in range(f_n)]
    f_num+=[-i-1 for i in range(f_n)]
    f_num=sorted(f_num)
    res=[]
    for i in range(len(f_num)):
        fnum=f_num[i]
        count=0+0j
        countlis=[]
        for j in range(len(lis)):
            weight=np.exp(-2*np.pi*fnum*(j/len(lis))*1j)
            count+=weight*lis[j]
            #print(count)
            countlis.append(count)
        #print(np.mean(countlis))
        res.append(count/len(lis))
    return res

def Dfourier(flis,rough=100):
    fn=len(flis)//2
    f_num=[0]
    f_num+=[i+1 for i in range(fn)]
    f_num+=[-i-1 for i in range(fn)]
    f_num=sorted(f_num)
    res=[]
    for i in range(rough):
        tmp=0+0j
        for j in range(len(f_num)):
            f_k=f_num[j]
            weight=np.exp(2*np.pi*f_k*(i/rough)*1j)
            tmp+=weight*flis[j]
        res.append(tmp)
    return res

def im2bin(im):
    res=np.zeros((len(im),len(im[0])))
    for i in range(len(im)):
        for j in range(len(im[i])):
            res[i][j]=0 if np.sum(im[i][j])>=600 else 1
    return res

def proc_contours(contours):
    res=[]
    for i in range(len(contours)):
        res.append(contours[i][0][1]+contours[i][0][0]*1j)
    return res

def contours2fourier(plis):
    f_s_lis=[]
    for i in range(len(plis)):
        contours_i=plis[i]
        contours_i=proc_contours(contours_i)
        f_s_tmp=fourier_fit(contours_i,len(contours_i)//2)
        f_s_lis.append(f_s_tmp)
    return f_s_lis

def proc_dflis(dflis):
    res=[]
    for i in range(len(dflis)):
        res.append([dflis[i].real,dflis[i].imag])
    return res

def fourier2contours(flis):
    res=[]
    for i in range(len(flis)):
        df=Dfourier(flis[i],len(flis[i])*2)
        res.append(proc_dflis(df))
    return res

def contours2im(contours,size):
    res=np.zeros(size)
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            point=contours[i][j]
            if point[0]>=size[0]-1:
                point[0]=size[0]-1
            if point[0]<0:
                point[0]=0
            if point[1]>=size[1]-1:
                point[1]=size[1]-1
            if point[1]<0:
                point[1]=0
            res[int(point[0])][int(point[1])]=1
    return res

def gen_circle(center,r):
    p_num=int(r*2*np.pi)
    plis=[]
    for i in range(p_num):
        point_tmp=r*np.exp(-2*np.pi*i*(1/p_num)*1j)
        point=[center[0]+point_tmp.real,center[1]+point_tmp.imag]
        plis.append(point)
    return plis
    

def play_anim(flis,size):
    res=[]
    contour_lis=[]
    for c in range(len(flis)):
        rough=int(len(flis[c]))
        fn=len(flis[c])//2
        f_num=[0]
        f_num+=[i+1 for i in range(fn)]
        f_num+=[-i-1 for i in range(fn)]
        f_num=sorted(f_num)
        flis[c]=[[f_num[i],flis[c][i]] for i in range(len(flis[c]))]
        flis[c]=sorted(flis[c],key=lambda x:abs(x[0]))
        flisnum=[i[0] for i in flis[c]]
        flis[c]=[i[1] for i in flis[c]]
        center_lis=[]
        
        for r in range(rough):
            plot_lis=[]
            for j in range(len(f_num)-1):
                f_ind=j
                f_nex_ind=j+1
                f_k=flis[c][f_ind]
                f_k_nex=flis[c][f_nex_ind]
                
                f_k=f_k
                '''if j!=0:
                    f_k_center=[f_k.real+center_lis[-1][0],f_k.imag+center_lis[-1][1]]
                    center_lis.append(f_k_center)
                else:
                    f_k_center=[f_k.real,f_k.imag]'''
                if j==0:

                    f_k_center=[f_k.real,f_k.imag]
                    center_lis.append(f_k_center)
                else:
                    weight=np.exp(2*np.pi*flisnum[j]*(r/rough)*1j)
                    f_pre=(flis[c][j])*weight+center_lis[-1][0]+center_lis[-1][1]*1j

                    f_k_center=[f_pre.real,f_pre.imag]
                    center_lis.append(f_k_center)
                if j==len(f_num)-2:
                    contour_lis.append(f_k_center)
                f_k_r=(f_k_nex.real**2+f_k_nex.imag**2)**0.5
                gc=gen_circle(f_k_center,f_k_r)
                plot_lis.append(gc)
            contours_im=contours2im([contour_lis],size)
            circle_im=contours2im(plot_lis,size)
            res.append(circle_im+contours_im)
            print('gen_anim:',c+1,':',len(flis),'channel',r+1,':',rough)
    return res