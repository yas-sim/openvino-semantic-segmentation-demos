import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

def main():

    ie = IECore()

    model='deeplabv3'   # in='mul_1/placeholder_port_1'(1,3,513,513), out='ArgMax/Squeeze'(1,513,513)
    model = './public/'+model+'/FP16/'+model
    net = ie.read_network(model+'.xml', model+'.bin')
    exec_net = ie.load_network(net, 'CPU')

    #cap = cv2.VideoCapture('movie1.264')
    cap = cv2.VideoCapture(0)

    bg = cv2.imread('background.jpg')
    ret, img = cap.read()
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

    key = -1
    while key!=27:
        ret, img = cap.read()
        if ret==False:
            return 0
        cv2.imshow('input', img)

        blob = cv2.resize(img, (513, 513))
        blob = blob.transpose((2,0,1))
        blob = blob.reshape((1,3,513,513))

        res = exec_net.infer(inputs={'mul_1/placeholder_port_1': blob}) ['ArgMax/Squeeze'][0]

        clsId = 15            # PascalVOC class  0=BG, 15=Person
        mask = np.where(res==clsId, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_p = np.reshape(mask, (img.shape[0], img.shape[1], 1))
        mask_n = 255-mask_p

        img    = img & mask_p
        bg_tmp = bg  & mask_n
        img    = img | bg_tmp 

        cv2.imshow('Result', img)
        key = cv2.waitKey(1)

if __name__ == '__main__':
    main()
