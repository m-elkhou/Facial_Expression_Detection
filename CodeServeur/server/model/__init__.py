import os, joblib, cv2, dlib
import socket
import uuid 

from model.feature import Feature
saved_img_path = 'server/static/result/'

ip = socket.gethostbyname(socket.gethostname())
url = 'http://'+ip+':'+str(os.getenv('PORT'))+'/static/result/'

model = joblib.load('model/tools/model.sav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/tools/shape_predictor_68_face_landmarks.dat")
ft = Feature(detector, predictor)
    
def draw_face_landmarks(img_path):
    img = cv2.imread(img_path)
    image = cv2.imread(img_path, 0) 
    detections = detector(image, 1)
    best_face = 0
          
    if len(detections) > 1:
        # On s'interisse au visage de grande dimension qui est le plus proche dans l'image       
        max_surface = 0
        xm, ym, wm, hm = 0, 0, 0, 0
        for i, face in enumerate(detections) :
            # Finding points for rectangle to draw on face
            xi, yi, wi,hi = face.left(), face.top(), face.width(), face.height()
            if wi*hi > max_surface:
                xm, ym, wm,hm = xi, yi, wi,hi
                max_surface = wi*hi
                best_face = i
    elif len(detections) < 1:
        return None
        
    for f, face in enumerate(detections) :
        shape = predictor(image, face) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(68): #Store X and Y coordinates in two lists
            xlist.append(int(shape.part(i).x))
            ylist.append(int(shape.part(i).y))

        def draw_line(x, y, g=True):
            if g :
                cv2.line(img, (xlist[x], ylist[x]), (xlist[y], ylist[y]), (0,255,0), 1)
            else:
                cv2.line(img, (xlist[x], ylist[x]), (xlist[y], ylist[y]), (155,0,247), 1)

        g=False 
        if f == best_face:
            g = True

        for i in range(16): draw_line(i,i+1, g) 
        for i in range(17, 21): draw_line(i,i+1, g)
        for i in range(22, 26): draw_line(i,i+1, g)
        for i in range(36, 41): draw_line(i,i+1, g)
        draw_line(36,41, g)
        for i in range(42, 47): draw_line(i,i+1, g)
        draw_line(42,47, g)

        for i in range(27, 35): draw_line(i,i+1, g)
        draw_line(30,35, g)

        for i in range(48, 59): draw_line(i,i+1, g)
        draw_line(48,59, g)
        for i in range(60, 67): draw_line(i,i+1, g)
        draw_line(60,67, g)
    
    # Drawing simple rectangle around found faces
    #cv2.rectangle(img, (xm, ym), (xm + wm, ym + hm), (0, 0, 255), 2)
	
	#generate rndom id for the of image
    uid=uuid.uuid1() #creating UUID
    uid_str=uid.urn #returns string value 'urn:uuid
    string_uid=uid_str[9:] #skip those 9 characters to get the string value
    img_name = string_uid+'.png' # img_path.split('/')[-1]
    cv2.imwrite(saved_img_path+img_name, img)
    print(saved_img_path+img_name)

    return img_name


def do_something(img_path):
    f = ft.get_feature(img_path, True)
    if f is None:
        return None, None,None
    # print(f.shape)

    emotion = ""
    pourcentage = ""
    for classe, pr in zip(model.classes_, model.predict_proba([f])[0]):
        emotion += classe+","
        pourcentage += str(int(pr*100))+","
    emotion = emotion[:-1]
    pourcentage = pourcentage[:-1]
    
    img_name = draw_face_landmarks(img_path)
    if img_name is None:
        return None, None,None

    # you must save img in this path 'server/static/result/'
    return emotion,pourcentage, url+img_name

def run(image_path):
    # u must return result as dictionary type
    print(image_path)
    status='200'

    emotion,pourcentage,url = do_something(image_path)
    if emotion is None:
        emotion = pourcentage = url = ''
        status='202'

    return {'status':status,'emotions':emotion,'pourcentages':pourcentage, 'url':url}