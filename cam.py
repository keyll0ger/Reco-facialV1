import cv2 as cv

dirCascadeFiles = r'opencv/data/haarcascades_cuda/'
#openCV : https://github.com/opencv/opencv/tree/3.4/data/haarcascades
classCascadefacial = cv.CascadeClassifier(dirCascadeFiles + "haarcascade_frontalface_default.xml")
 
def facialDetectionAndMark(_image, _classCascade):
    imgreturn = _image.copy()
    gray = cv.cvtColor(imgreturn, cv.COLOR_BGR2GRAY)
    faces = _classCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv.rectangle(imgreturn, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return imgreturn

	
def videoDetection(_haarclass):
    webcam = cv.VideoCapture(0)
    if webcam.isOpened():
        while True:
            bImgReady, imageframe = webcam.read() 
            if bImgReady:
                face = facialDetectionAndMark(imageframe, _haarclass)
                cv.imshow('My webcam', face) 
            else:
                print('No image available')
            keystroke = cv.waitKey(20) 
            if (keystroke == 27):
                break 
 
        webcam.release()
        cv.destroyAllWindows()   
         
videoDetection(classCascadefacial)
