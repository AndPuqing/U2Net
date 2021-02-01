import cv2

cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    xmlfile = r'D:\Softwares\Anaconda\envs\tensorflow\Lib\site-packages\cv2\data\haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(xmlfile)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
    )
    print("发现{0}个目标!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
