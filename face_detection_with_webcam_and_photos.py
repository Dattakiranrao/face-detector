import cv2 as cv 

trained_faces_data = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')

print("Welcome to Face Detection App ")
print("If you want to quit the application pleas press (q)")
choise = input("Press (w) for webcam or (p) for testin with photos :  ")

w, p = 'w', 'p'

if choise == p:
    img = input("enter the image name and add it to current working directory :  ")
    image = cv.imread(img)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    face_rectangle_coordinates = trained_faces_data.detectMultiScale(gray_image)

    for (x, y, w, h) in face_rectangle_coordinates:
        cv.rectangle(image, (x, y), (x+w , y+h), (0,0,255), 4)

    cv.imshow('Image', image) 
    cv.waitKey()

elif choise == w:
    webcam = cv.VideoCapture(0)

    while True:
        read_status, frame = webcam.read()
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        face_rectangle_coordinates = trained_faces_data.detectMultiScale(frame)

        for (x, y, w, h) in face_rectangle_coordinates:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

        cv.imshow("Frames" , frame)
        key = cv.waitKey(1)

        if key == 113 or key == 81:
            break
    
    webcam.release()

