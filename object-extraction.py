import random

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

class splite:
    img_h = 0
    img_w = 0
    def __init__(self, image, mask, x, y, w, h):
        self.image = image
        self.mask = mask
        self.x=int(x)
        self.y=int(y)
        self.w=int(w)
        self.h=int(h)
        self.xs = random.random()*4.-2.
        self.ys = random.random()*4.-2.
        self.xs = 0
        self.ys = 0

        self.x0 = float(x)
        self.y0 = float(y)

    def move(self):
        global g_img_H
        global g_img_W
        self.x0 = self.x0+self.xs
        self.y0 = self.y0+self.ys
        if self.x0<0:
            self.x0 = -self.x0
            self.xs = -self.xs
        if self.y0<0:
            self.y0 = -self.y0
            self.ys = -self.ys
        if self.x0+self.w>splite.img_w:
            self.x0 = self.x0 - (self.x0+self.w-splite.img_w)
            self.xs = -self.xs
        if self.y0+self.h>splite.img_h:
            self.y0 = self.y0 - (self.y0+self.h-splite.img_h)
            self.ys = -self.ys
        self.x = int(self.x0)
        self.y = int(self.y0)

    def set(self, x, y):
        self.x0 = x
        self.y0 = y
        if self.x0<0: self.x0=0
        if self.y0<0: self.y0=0
        if self.x0+self.w>splite.img_w: self.x0=splite.img_w-self.w
        if self.y0+self.h>splite.img_h: self.y0=splite.img_h-self.h
        self.x = int(self.x0)
        self.y = int(self.y0)


g_grabbedSplite = -1
g_prevX = -1
g_prevY = -1

def onMouse(event, x, y, flags, splites):
    global g_grabbedSplite
    global g_prevX, g_prevY
    dx = x-g_prevX
    dy = y-g_prevY
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, sp in enumerate(splites):
            if x>=sp.x and x<sp.x+sp.w and y>=sp.y and y<sp.y+sp.h:
                g_grabbedSplite = i
                sp.xs=0
                sp.ys=0
    if event == cv2.EVENT_MOUSEMOVE:
        if g_grabbedSplite !=-1:
            sid = g_grabbedSplite
            splites[sid].set(splites[sid].x+dx, splites[sid].y+dy)
    if event == cv2.EVENT_LBUTTONUP:
        if g_grabbedSplite !=-1:
            sid = g_grabbedSplite
        g_grabbedSplite = -1
    g_prevX = x
    g_prevY = y


def objectExtraction(img, clsId, exec_net_seg, exec_net_inp):

    blob = cv2.resize(img, (513, 513))
    blob = blob.transpose((2,0,1))
    blob = blob.reshape((1,3,513,513))

    res_seg = exec_net_seg.infer(inputs={'mul_1/placeholder_port_1': blob}) ['ArgMax/Squeeze'][0]

    # create mask images (mask_inp: mask for inpainting(0.0-1.0),  mask_img:mask for inpaint input image(unit8), mask_C1:mask in the same size as original image(uint8) )
    mask_inp = np.where(res_seg==clsId, 255, 0).astype(np.uint8)
    mask_inp = np.reshape(mask_inp, (513,513,1))                        # Reshape for resizing
    mask_C1  = cv2.resize(mask_inp, (img.shape[1], img.shape[0]))       # The segmentation mask in the same size as input image (h,w,1)
    mask_inp = cv2.resize(mask_inp, (680,512)  , interpolation=cv2.INTER_LINEAR)
    mask_img = np.where(mask_inp>0, 255, 0).astype(np.uint8)            # For masking the input image
    mask_inp = np.where(mask_inp>0, 1., 0.).astype(np.float32)
    mask_inp = np.reshape(mask_inp, (1, 512,680))

    img_inp  = cv2.resize(img , (680,512), interpolation=cv2.INTER_LINEAR)
    mask = cv2.merge([mask_img, mask_img, mask_img])
    img_inp |= mask                                                     # Mask input image with the segmentation result
    img_inp  = img_inp.transpose((2,0,1))
    img_inp  = img_inp.reshape((1,3,512,680))

    res_inp = exec_net_inp.infer(inputs={'Placeholder': img_inp, 'Placeholder_1': mask_inp}) ['Minimum']

    res_inp = np.transpose(res_inp, (0, 2, 3, 1)).astype(np.uint8)
    res_inp = cv2.cvtColor(res_inp[0], cv2.COLOR_RGB2BGR)
    res_inp = cv2.resize(res_inp, (img.shape[1], img.shape[0]))

    # find bounding boxes of objects found and crop objects
    contours, hierarchy = cv2.findContours(mask_C1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_C1 = np.where(mask_C1>0, 255, 0).astype(np.uint8)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_C1 = cv2.dilate(mask_C1, kernel, 3)                        # Dilate the mask a little
    mask_C3 = cv2.merge([mask_C1, mask_C1, mask_C1])
    splites=[]
    for contour in contours:
        if len(contour)==0: continue
        if cv2.contourArea(contour)<50: continue
        x, y, w, h = cv2.boundingRect(contour)
        splite_img = img[y:y+h, x:x+w]
        splite_msk = mask_C3[y:y+h, x:x+w]
        splite_img &= splite_msk
        splites.append(splite(splite_img, splite_msk, x, y, w, h))

    cv2.setMouseCallback('Result', onMouse, splites)

    key = -1
    count = 0
    while key!=27 and key!=ord(' '):
        #tmpimg = img.copy()
        tmpimg = res_inp.copy()
        msk_p  = np.zeros((img.shape), dtype=np.uint8)
        img_sp = np.zeros((img.shape), dtype=np.uint8)
        for sp in splites:
            sp.move()
            msk_p [sp.y:sp.y+sp.h, sp.x:sp.x+sp.w] |= sp.mask
            img_sp[sp.y:sp.y+sp.h, sp.x:sp.x+sp.w] |= sp.image
            if count==0:
                img_sp[sp.y:sp.y+sp.h, sp.x:sp.x+sp.w] |= sp.mask
        msk_n = 255-msk_p
        tmpimg &= msk_n
        tmpimg |= img_sp

        cv2.imshow('Result', tmpimg)
        key = cv2.waitKey(1000//30)
        count = (count+1) % 30

    return



def main():

    ie = IECore()

    model='deeplabv3'   # in='mul_1/placeholder_port_1'(1,3,513,513), out='ArgMax/Squeeze'(1,513,513)
    model = './public/'+model+'/FP16/'+model
    net_seg = ie.read_network(model+'.xml', model+'.bin')
    exec_net_seg = ie.load_network(net_seg, 'CPU')

    model='gmcnn-places2-tf' # in1='Placeholder'(1,3,512,680), in2='Placeholder_1'(1,1,512,680), out='Minimum'(1,3,512,680)
    model = './public/'+model+'/FP16/'+model
    net_inp = ie.read_network(model+'.xml', model+'.bin')
    exec_net_inp = ie.load_network(net_inp, 'CPU')

    cv2.namedWindow('Result')

    #cap = cv2.VideoCapture('movie4.264')
    cap = cv2.VideoCapture(0)

    _, img = cap.read()
    splite.img_h = img.shape[0]
    splite.img_w = img.shape[1]

    key = -1
    while key!=27:
        ret, img = cap.read()
        if ret==False:
            return 0

        if key==ord(' '):
            clsId = 15            # PascalVOC class  0=BG, 15=Person
            objectExtraction(img, clsId, exec_net_seg, exec_net_inp)

        cv2.imshow('Result', img)
        key = cv2.waitKey(30)

if __name__ == '__main__':
    main()
