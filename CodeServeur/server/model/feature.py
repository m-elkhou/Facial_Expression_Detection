import cv2
import numpy as np
import dlib, math
from skimage import feature
from skimage.feature import hog

class Feature:
    def __init__(self, detector, predictor):
        home = 'model/tools/'
        self.faceDet  = cv2.CascadeClassifier(home+"haarcascade_frontalface_default.xml")
        self.faceDet2 = cv2.CascadeClassifier(home+"haarcascade_frontalface_alt2.xml")
        self.faceDet3 = cv2.CascadeClassifier(home+"haarcascade_frontalface_alt.xml")
        self.faceDet4 = cv2.CascadeClassifier(home+"haarcascade_frontalface_alt_tree.xml")
        self.detector = detector
        self.predictor = predictor
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def face_extractor(self, img_path):
        gray = cv2.imread(img_path,0)
        # Detection des visages : il peut detecter des données aberantes(un objet n'est pas clair peut etre considerer
        # comme un visage)
        faces = self.faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) < 1:
            faces = self.faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)
            print(2)
            if len(faces) < 1:
                print(3)
                faces = self.faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) < 1:
                    print(4)
                    faces = self.faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(faces) < 1:
                        return None
        # On s'intersse au visage de grande dimension qui est le plus proche dans l'image       
        max_surface = 0
        x, y, w,h = 0, 0, 0, 0
        for (xi, yi, wi,hi) in faces:
            if wi*hi > max_surface:
                x, y, w,h = xi, yi, wi,hi
                max_surface = wi*hi
        if(w*h!=0):
            #si h = 0 ou w = 0 (rien detecté), notre fonction retourne None,
            # sinon on bien determiner le visage donc on l'extract selon le nouveau hauteur et largeur
            face_seg = gray[y:y+h, x:x+w]
            # on fait la normalisation de hauteur et de largeur  des images h=96 et w=96
            face_seg = cv2.resize(face_seg, (96, 96), interpolation = cv2.INTER_AREA) # Resize face so all images have same size
            return face_seg
        
        return None

    def our_ft_landmark(self, X, Y):
        """ 
        Dans cette fonction on a utilisé les points de landmark pour l'extraction features cela on calculant des ongles 
        entre les points des parties suivantes: entre le nez et les yeux, les yeux et les sourcils et la bouche et le nez ... 
        Apres ce calcul on fait la normalisation de données pour gerer les problemes liées à la difference des visages humains.
        """
        def get_degre(i,j):
            myradians = math.atan2(Y[i]-Y[j], X[i]-X[j])
            return math.degrees(myradians)

        f_x = [30,30,30,30,30,30,21,17,22,43,42,38,36,62,51,48,51]
        f_y = [26,54,17,48,22,21,22,21,26,47,45,40,39,66,57,54,30]
        features = []

        for i,j in zip(f_x,f_y):
            features.append(np.linalg.norm( (X[i]-X[j],Y[i]-Y[j]) ) )

        features = list(features / np.mean(features))# normalisation des données
        
        features2 = []
        features2.append((get_degre(30,26)-90))
        features2.append((get_degre(30,22)-90))
        features2.append((90-get_degre(30,17)))
        features2.append((90-get_degre(30,21)))
        features2.append((get_degre(30,54)*-1-90))
        features2.append((90-get_degre(30,48)*-1))

        features.extend(np.array(features2)/100)
        return features

    def get_landmarks(self, image): # from https://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
        """
        Dans cette fonction on recupere les points de landmarks et les features (our_ft_landmark) 
        """
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        detections = self.detector(image, 1)
        if len(detections) < 1: # Number of faces detected = 0
            # print("Number of faces detected: {}".format(len(detections)))
            return None
        # Draw Facial Landmarks with the predictor class
        shape = self.predictor(image, detections[0])
        xlist = []
        ylist = []
        for i in range(68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        landmarks_vectorised = []
        landmarks_vectorised = self.our_ft_landmark(xlist, ylist)# Extaraction des features

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
    #       landmarks_vectorised.append(x)
    #       landmarks_vectorised.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)# Distance euclidienne
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))# Calcule de l'ongle entre le moyenne et un point

        return landmarks_vectorised
    
    def get_face_landmarks(self, img_path):
        """
        Une  autre façon de detecter le visage.
        """
        image = cv2.imread(img_path, 0)
        try:
            detections = detector(image, 1)
        except:
            return None
        if len(detections) < 1:
            return None
        # Draw Facial Landmarks with the predictor class
        shape = self.predictor(image, detections[0])
        xlist = []
        ylist = []
        for i in range(68):  # Store X and Y coordinates in two lists
            xlist.append(int(shape.part(i).x))
            ylist.append(int(shape.part(i).y))
        # Estimation de visage
        xmin, xmax, ymin, ymax = np.min(xlist), np.max(
            xlist), np.min(ylist), np.max(ylist)
        w, h = xmax-xmin, ymax-ymin
        if ymin <= h/3:
            ymin = 0
        else:
            ymin -= int(h/3)

        if xmin <= w/5:
            xmin = 0
        else:
            xmin -= int(w/5)

        hight, wight = image.shape
        if ymax >= hight+h/5:
            ymax = hight
        else:
            ymax += int(h/5)

        if xmax >= wight-w/5:
            xmax = wight
        else:
            xmax += int(w/5)

        face_seg = image[ymin:ymax, xmin:xmax]
        face_seg = cv2.resize(face_seg, (96, 96), interpolation = cv2.INTER_AREA)
        return face_seg


    def get_hog(self, img):
        # Extraction de features en utilisant Histogram of oriented gradient
        return hog(img, orientations=6, pixels_per_cell=(9, 9), cells_per_block=(1, 1))

    def sliding_hog_windows(self, image):
        """
        Extraction de features en utilisant Hog avec la methode sliding window
        Pour chaque fenetre, on le superposer avec chaque region de notre image afin de detecter l'objet qui nous interesse.
        """
        # initialization
        image_height, image_width  = 48, 48
        window_size = 24
        window_step = 6
        hog_windows = []
        for y in range(0, image_height, window_step):
            for x in range(0, image_width, window_step):
                window = image[y:y+window_size, x:x+window_size]
                hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                                cells_per_block=(1, 1)))
        return hog_windows
    
    def build_filters(self):
        filters = []
        ksize = 9
        #define the range for theta and nu
        for theta in np.arange(0, np.pi, np.pi / 8):
            for nu in np.arange(0, 6*np.pi/4 , np.pi / 4):
                kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, nu, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
        return filters

    #---------------------------------------------------
    #function to convolve the image with the filters
    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def extracting_features_gabor_filter_bank(self, imgg):
        #instantiating the filters
        filters = self.build_filters()
        f = np.asarray(filters)
        feat = []
        #calculating the local energy for each convolved image
        for j in range(40):
            res = self.process(imgg, f[j])
            temp = 0
            for p in range(imgg.shape[0]):
                for q in range(imgg.shape[1]):
                    temp = temp + res[p][q]*res[p][q]
            feat.append(temp)
        #calculating the mean amplitude for each convolved image	
        for j in range(40):
            res = self.process(imgg, f[j])
            temp = 0
            for p in range(imgg.shape[0]):
                for q in range(imgg.shape[1]):
                    temp = temp + abs(res[p][q])
                feat.append(temp)
        #feat matrix is the feature vector for the image
        #print(feat)
        return feat

    def get_feature(self, img_path, face_cut=False):
        # Tous les visages sont de dimension = (96,96)
        if face_cut:
            img = self.face_extractor(img_path)
        else:
            try:
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
            except:
                print("tswira srira bzzaf !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                return None
            

        if img is None:
            img = self.get_face_landmarks(img_path)
            if img is None:
                return None

        img = self.clahe.apply(img)
        ft_landmarks = self.get_landmarks(img.copy()) # len = 295

        if ft_landmarks is None:
            return None
        # print("LN", len(ft_landmarks))
        try:
            img = cv2.resize(img, (48, 48), interpolation = cv2.INTER_AREA) # Pour reduire features de Hog
        except:
            return None

        ft_hog = self.get_hog(img.copy()) # len = 150
        # print("HOG", len(ft_hog))

        ft_shogw = self.sliding_hog_windows(img.copy()) # len = 2592
        # print("HOG_SW", len(ft_shogw))
        ft_gabor = self.extracting_features_gabor_filter_bank(img.copy()) # len= 1960
        # print("GABOR", len(ft_gabor))
        global_feature = np.concatenate([ft_landmarks, ft_gabor, ft_hog, ft_shogw]).flatten()

        return global_feature