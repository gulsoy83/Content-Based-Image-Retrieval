import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import sys
from math import sqrt


path1 = "elephant/*.jpg"
path2 = "flamingo/*.jpg"
path3 = "kangaroo/*.jpg"
path4 = "leopards/*.jpg"
path5 = "octopus/*.jpg"
path6 = "sea_horse/*.jpg"

images=[]
t_images = []
sayac=0;
for file in glob.glob(path1):
    if sayac >=30:
        break
    elif sayac >= 20:
        t_images.append( cv2.imread(file))
        sayac = sayac +1
    else:
        images.append(cv2.imread(file))
        sayac = sayac +1
sayac=0;
for file in glob.glob(path2):
    if sayac >=30:
        break
    elif sayac >= 20:
        t_images.append( cv2.imread(file))
        sayac = sayac +1
    else:
        images.append(cv2.imread(file))
        sayac = sayac +1
sayac=0;
for file in glob.glob(path3):
    if sayac >=30:
        break
    elif sayac >= 20:
        t_images.append( cv2.imread(file))
        sayac = sayac +1
    else:
        images.append(cv2.imread(file))
        sayac = sayac +1
sayac=0;
for file in glob.glob(path4):
    if sayac >=30:
        break
    elif sayac >= 20:
        t_images.append( cv2.imread(file))
        sayac = sayac +1
    else:
        images.append(cv2.imread(file))
        sayac = sayac +1
sayac=0;
for file in glob.glob(path5):
    if sayac >=30:
        break
    elif sayac >= 20:
        t_images.append( cv2.imread(file))
        sayac = sayac +1
    else:
        images.append(cv2.imread(file))
        sayac = sayac +1
sayac=0;
for file in glob.glob(path6):
    if sayac >=30:
        break
    elif sayac >= 20:
        t_images.append( cv2.imread(file))
        sayac = sayac +1
    else:
        images.append(cv2.imread(file))
        sayac = sayac +1


    
    



def cal_hist(img):
    hist = [[0 for x in range(256)] for y in range(4)]
    hist[0] = np.zeros(shape = (256))
    hist[1] = np.zeros(shape = (256))
    hist[2] = np.zeros(shape = (256))

    hist[3] = np.zeros(shape = (256))

    (h,w, c) = img.shape

    coz = h*w
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for i in range(w):
        for j in range(h):
            hist[0][img[j,i,0]] +=1
            hist[1][img[j,i,1]] +=1
            hist[2][img[j,i,2]] +=1

            hist[3][img_hsv[j,i,0]]  +=1

    for i in range(256):
        hist[0][i] = hist[0][i] /coz
        #hist[0][i] = round (hist[0][i], 5)
            
        hist[1][i] = hist[1][i]  /coz
        #hist[1][i] = round (hist[1][i], 5)
        
        hist[2][i] = hist[2][i] /coz
        #hist[2][i] = round (hist[2][i], 5)
        hist[3][i] = hist[3][i] /coz
        #print("{:f}".format(hist[0][i]))
    return hist



all_hist=[]

for i in range(len(images)):
    img_bgr = images[i]
    #b, g, r = img_bgr[:,:,0], img_bgr[:,:,1], img_bgr[:,:,2]
    
    #img_hsv = cv2.cvtColor(images1[19],cv2.COLOR_BGR2HSV)
    #h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

    
    sonuc = cal_hist(img_bgr)
    all_hist = all_hist + [sonuc]
    #plt.plot(all_hist[i][0], color = 'b')
    #plt.plot(all_hist[i][1], color = 'g')
    #plt.plot(all_hist[i][2], color = 'r')
    #plt.plot(all_hist[i][3], color = 'k')
    #plt.show()

print("Training tamamlandi.")


test_hist=[]
for i in range(len(t_images)):
    img_bgr = t_images[i]
    sonuc = cal_hist(img_bgr)
    test_hist = test_hist + [sonuc]

   
print("Test resimlerinin histogramlari hesaplandi.")

def distance3(R1, R2):
    blue = R1[0]
    green = R1[1]
    red = R1[2]
    
    tblue = R2[0]
    tgreen = R2[1]
    tred = R2[2]
    bsum =0
    rsum =0
    gsum =0
    for i in range(256):
        bsum += (blue[i] - tblue[i]) ** 2
        gsum += (green[i] - tgreen[i]) **2
        rsum += (red[i] - tred[i]) **2
    bdistance = sqrt(bsum)
    gdistance = sqrt(gsum)
    rdistance = sqrt(rsum)

    dist = sqrt( bdistance**2 + gdistance**2 + rdistance**2 )

    
    return dist

def distance1 (R1, R2):
    h = R1[0]
    th = R2[0]
    hsum = 0
    for i in range(256):
        hsum += (h[i]-th[i]) ** 2
    hdist = sqrt(hsum)
    return hdist

eu = []
for i in range (len(test_hist)):
    for j in range(len(all_hist)):

        
        
        V1 = [ all_hist[j][0], all_hist[j][1], all_hist[j][2] ]
        V2= [ all_hist[j][3]]

        T1 = [ test_hist[i][0], test_hist[i][1], test_hist[i][2] ]
        T2 = [test_hist [i][3]]
        
        bgruzaklik = distance3(V1,T1)
        huzaklik = distance1(V2,T2)

        eu.append( (i,j,bgruzaklik,huzaklik))
        

eu.sort(key=lambda x: x[2])
testindis = 0
while testindis<len(t_images) :
    sayac = 0
    for i in range(len(eu)):
        if(eu[i][0] == testindis):
            if sayac<5:
                orad = "rgborg"
                orad += str(testindis)
                ad = "rgbbenzer"
                ad += str(sayac)
                cv2.imshow(orad, t_images[eu[i][0]])
                cv2.imshow(ad, images[eu[i][1]])
                
                
                sayac = sayac +1
        if sayac>4:
            break
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.destroyAllWindows()
        testindis = testindis +1
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
        
        
eu.sort(key=lambda x: x[3])
testindis = 0
while testindis < len(t_images):
    sayac = 0
    for i in range(len(eu)):
        if(eu[i][0] == testindis):
            if sayac<5:
                orad = "horg"
                orad += str(testindis)
                ad = "hbenzer"
                ad += str(sayac)
                cv2.imshow(orad, t_images[eu[i][0]])
                cv2.imshow(ad, images[eu[i][1]])
                sayac = sayac +1
        if sayac>4:
            break
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.destroyAllWindows()
        testindis = testindis +1
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break


sys.exit()

