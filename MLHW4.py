import numpy as np
from scipy.misc.pilutil import imresize
import cv2 
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

DIGIT_WIDTH = 10 
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10 # 0-9


# Support Vector Machine, dati il training l'algoritmo genera un iperpiano ottimale che classifica nuovi esempi
# Cerchiamo l'iperpiano con distanza minima maggiore rispetto agli esempi di addestramento (Massimizza il margine dei dati di allenamento)

class NeuralNetwork_SVM():
    def __init__(self, num_feats, C = 1, gamma = 0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF) #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1,self.features))
        return results[1].ravel()



# Classifica gli oggetti basandosi sulle caratteristiche degli oggetti vicini a quello analizzato, 
# assegnandolo ad una classe se questa e' la piu' frequenta fra i k esempi piu' vicini a lui

class NeuralNetwork_KNearest():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()



class MLHW4():

    # Questo metodo suddivide l'input training image in piccole celle (di una singola cifra) e utilizza queste celle come dati per il training.
    # La default training image (MNIST) e' di dimensioni 1000x1000 ed ogni numero e' 10x20. Quindi dividiamo 1000/10 orizontalmente e 1000/20 verticalmente.
    def split(self, img, cell_size, flatten=True):
        h, w = img.shape[:2]
        sx, sy = cell_size
        cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
        cells = np.array(cells)
        if flatten:
            cells = cells.reshape(-1, sy, sx)
        return cells

    def loadTraining(self, fn):
        print('\n 1) Loading "%s for training" ...' % fn)
        digits_img = cv2.imread(fn, 0)
        digits = self.split(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
        resized_digits = []
        for digit in digits:
            resized_digits.append(imresize(digit,(IMG_WIDTH, IMG_HEIGHT)))
        labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
        return np.array(resized_digits), labels

    def pixelstohog20(self, img_array):
        hog_featuresData = []
        for img in img_array:
            fd = hog(img, 
                     orientations=10, 
                     pixels_per_cell=(5,5),
                     cells_per_block=(1,1), 
                     visualise=False)
            hog_featuresData.append(fd)
        hog_features = np.array(hog_featuresData, 'float64')
        return np.float32(hog_features)



    def getdigits(self, contours, hierarchy):
        hierarchy = hierarchy[0]
        bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
        final_bounding_rectangles = []
        #find the most common heirarchy level - that is where our digits's bounding boxes are
        u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
        most_common_heirarchy = u[np.argmax(np.bincount(indices))]
        
        for r,hr in zip(bounding_rectangles, hierarchy):
            x,y,w,h = r
            # queesto puo' variare in base all'immagine passata in input che si chiede di predire
            # stiamo cercando di estrarre SOLO i rettangoli con le immagini (questo e' un modo molto semplice per farlo).
            # utilizziamo l'heirarchy per estrarre solo le caselle che si trovano nello stesso livello globale - per evitare cifre all'interno di altre cifre
            # ex: potrebbe esserci un riquadro di delimitazione all'interno di ogni 6,9,8 a causa dei loop nell'aspetto del numero - non lo vogliamo.
            # per saperne di piu': https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
            if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
                final_bounding_rectangles.append(r)    

        return final_bounding_rectangles


    def CreateOutPutFile(self, img_file, model):


        im = cv2.imread(img_file)    
        blank_image = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
        blank_image.fill(255)

        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        plt.imshow(imgray)
        kernel = np.ones((5,5),np.uint8)
        
        ret,thresh = cv2.threshold(imgray,127,255,0)   
        thresh = cv2.erode(thresh,kernel,iterations = 1)
        thresh = cv2.dilate(thresh,kernel,iterations = 1)
        thresh = cv2.erode(thresh,kernel,iterations = 1)
        
        _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        digits_rectangles = self.getdigits(contours,hierarchy)  # rettangoli di delimitazione delle cifre nell'immagine dell'utente
        
        for rect in digits_rectangles:
            x,y,w,h = rect
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            im_digit = imgray[y:y+h,x:x+w]
            im_digit = (255-im_digit)
            im_digit = imresize(im_digit,(IMG_WIDTH ,IMG_HEIGHT))

            hog_img_data = self.pixelstohog20([im_digit])  
            pred = model.predict(hog_img_data)
            cv2.putText(im, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv2.putText(blank_image, str(int(pred[0])), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)


        plt.imshow(im)
        cv2.imwrite("output/comparison_"+img_file,im) 
        cv2.imwrite("output/digital_conversion_"+img_file,blank_image) 
        cv2.destroyAllWindows()           



    def Start(self,FileTraining,FileTest):

        digits, labels = self.loadTraining(FileTraining) 

        print('Train: ',digits.shape)
        print('Test: ',labels.shape)

        digits, labels = shuffle(digits, labels, random_state=256)
        train_digits_data = self.pixelstohog20(digits)
        X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)


        print("\n 2) Training K-Nearest Neighbor")
        model = NeuralNetwork_KNearest(k = 3)
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        print('Accuracy: ',accuracy_score(y_test, preds))


        print("\n 3) Training Support Vector Machine")
        model = NeuralNetwork_SVM(num_feats = train_digits_data.shape[1])
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        print('Accuracy: ',accuracy_score(y_test, preds))


        print("\n 4) Generate Files")
        self.CreateOutPutFile(FileTest, model)



