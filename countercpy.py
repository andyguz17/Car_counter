#Counting
import cv2
import numpy as np
import time 
import cvmath as cvm
r = [] 
x = []
y = []
w = []
h = []
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture('traffic1.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=10, backgroundRatio=0.7, noiseSigma=0)
#fgbg = cv2.createBackgroundSubtractorMOG2()
cv2.ocl.setUseOpenCL(False)
counter = 0
cars = 0
ca = 0
tx = []
ty = []
ctr1 = 0
ctr2 = 0
n = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1080,720))
    if (counter == 60):
        nbox=input("Número de áreas de interés: ")
        nbox=int(nbox)

        fromCenter = False
        img = frame.copy()


        for i in range (nbox):
            cv2.putText(img, "Select your Roi # "+str(i+1) ,(200,40), font, 1,(150,150,58),2,cv2.LINE_AA)
            m = cv2.selectROI(img, fromCenter) 
            m = np.array(m)
            r.append(m)
            img = frame.copy()

        cv2.destroyAllWindows()
        
        r=np.array(r)
        print(r)
        for i in range(nbox):
            x.append(r[i,0])
            y.append(r[i,1])
            w.append(r[i,2])
            h.append(r[i,3])
        x=np.array(x)
        y=np.array(y)
        w=np.array(w)
        h=np.array(h)    
        for i in range (nbox):
            cv2.rectangle(frame,(int(x[i]),int(y[i])),(int(x[i])+int(w[i]),int(y[i])+int(h[i])),(57,0,199),2)
            cv2.line(frame,(int(x[i]),int(y[i]+(h[i]/2))),(int(x[i]+w[i]),int(y[i]+(h[i]/2))),(0,195,255),3)
        cv2.imshow('frame',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    elif( counter >= 90):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        kernel = np.ones((1,1),np.uint8)
        kernel2 = np.ones((6,6),np.uint8)
        kernel3 = np.ones((4,4),np.uint8)
        
        fgmask = fgbg.apply(gray)
        
        op = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        
        ret,th1 = cv2.threshold(op,200,255,cv2.THRESH_BINARY)

        cl = cv2.morphologyEx(th1,cv2.MORPH_CLOSE,kernel2)
        cl = cv2.dilate(cl,kernel3,iterations = 1)
        
        #median = cv2.medianBlur(cl, 3)
        fil = cv2.filter2D(cl,-1,kernel)

        mask= np.zeros_like(fil)

        for i in range (nbox):
            cv2.rectangle(mask,(int(x[i]),int(y[i])),(int(x[i])+int(w[i]),int(y[i])+int(h[i])),(57,0,199),-1)
            cv2.line(frame,(int(x[i]),int(y[i]+(h[i]/2))),(int(x[i]+w[i]),int(y[i]+(h[i]/2))),(0,195,255),3)
        masked = cv2.bitwise_and(fil,mask)

        contornosimg = masked.copy()

        im,contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE   ,cv2.CHAIN_APPROX_SIMPLE)

        image = frame.copy()
        for i,rect in enumerate(contornos):
            # Eliminamos los contornos más pequeños
            if cv2.contourArea(rect) < 2700:
                    continue

            (a,b,c,d) = cv2.boundingRect(rect)

            # Dibujamos el rectángulo del bounds
            cv2.rectangle(image, (a, b), (a + c, b + d), (0, 255, 0), 1)
            
            m=int(c/2)
            n=int(d/2)
            xa = a+m
            ya = b+n
            tx.append(xa)
            ty.append(ya)
            if (i >= n):
                n=i
            else: 
                n=n 

        tx=np.array(tx)           
        ty=np.array(ty)

        if (counter == 95):
            tpx = np.copy(tx)
            tpy = np.copy(ty)
            
            tp = frame.copy()
            
            mask = tp    
            res = tp
            th = tp
        elif(counter > 95):
            for k in range (nbox):
                for i in range(ty.shape[0]):
                    for j in range(tpy.shape[0]):
                        if (abs(tpx[j]-tx[i])<=50):
                            if(abs(tpy[j]-ty[i])<=50):
                                lower_red = np.array([255,0,255])
                                upper_red = np.array([255,0,255])
                                
                                cv2.line(tp,(tx[i],ty[i]),(tpx[j],tpy[j]),(255,0,255),5)
                                L2 =cvm.line(cvm.dot(tx[i],ty[i]),cvm.dot(tpx[j],tpy[j]))
                            
                                if (tpy[j]>ty[i]):
                                    for i2 in range (ty[i],tpy[j]):
                                        if (i2==int(y[k]+(h[k]/2))):
                                            cv2.line(image,(int(x[k]),int(y[k]+(h[k]/2))),(int(x[k]+w[k]),int(y[k]+(h[k]/2))),(0,0,255),3)
                                            ctr1+=1
                                            print("Arriba:   "+str(ctr1))
                                elif(ty[i]>tpy[j]):
                                    for i2 in range(tpy[j],ty[i]):
                                        if (i2==int(y[k]+(h[k]/2))):
                                            cv2.line(image,(int(x[k]),int(y[k]+(h[k]/2))),(int(x[k]+w[k]),int(y[k]+(h[k]/2))),(0,0,255),3)
                                            ctr2+=1
                                            print("Abajo:   "+str(ctr2))
                                
                                mask = cv2.inRange(tp, lower_red, upper_red)
                                res = cv2.bitwise_and(tp,tp, mask= mask)
                                
                                #image = image +res
                                image = cv2.addWeighted(image,1, res, 0.8, 0.1) 
                        else:
                            tp=frame.copy()

        tpx = np.copy(tx)
        tpy = np.copy(ty)


        tx=[]
        ty=[]

        cv2.imshow('Filtrada', masked)
        cv2.imshow('Original',image )
        cv2.imshow('Contornos',contornosimg)

        k=cv2.waitKey(30) & 0xff
        if (k==27):
            break

    counter+=1
cap.release()
cv2.destroyAllWindows()