# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:43:48 2023

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
            
#读取图片
im=cv2.imread('img/changan.jpg')
#图片二值化
im_bin=im2bin(im).astype(np.uint8)
#寻找轮廓
contours,hier=cv2.findContours(im_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#轮廓转换成傅里叶级数
fslis=contours2fourier(contours)
#傅里叶级数转换成轮廓
dflis=fourier2contours(fslis)
#轮廓生成图像
dfim=contours2im(dflis,(len(im_bin),len(im_bin[0])))
plt.figure(0)
plt.imshow(dfim)
#生成动画
anim=play_anim(fslis,(len(im_bin),len(im_bin[0])))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('pic_anim.avi', fourcc, fps=50, frameSize=(len(im_bin[0]),len(im_bin)))
for i in range(len(anim)):
    cv2.imshow('circle',anim[i])
    cv2.waitKey(10)
    
    anim_i=anim[i].astype(np.uint8)
            
    anim_i = np.stack([anim_i,anim_i,anim_i],axis=-1)*255#cv2.cvtColor(anim_i, cv2.COLOR_GRAY2BGR)*255
    #print(anim_i[150,150])
    out_video.write(anim_i)
out_video.release()
cv2.destroyAllWindows()
print('out ok')