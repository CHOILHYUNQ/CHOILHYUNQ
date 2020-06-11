import cv2
import pymysql.cursors


def Load_LabelIDs(fn): #메모장
    labelNames = [ "hyojin","illhyun","minjae","taewon"]

    with open(fn) as f:
        data = f.readlines()
        for i in range(len(data)):
            lines = str(data[i]).split("\\n")
            for s in lines:
              labelNames.append(s)
    return labelNames

labelNames = Load_LabelIDs('labelIDs.txt')
labelDics = {}
for s in labelNames:
    strs = str(s).split("=")
    labelDics[strs[0]] = strs[0].split("\n")[0]
    print(str(s)) #3
face_cascade = cv2.CascadeClassifier('haarcascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")  # 저장된 값 가져오기

cap = cv2.VideoCapture(0)  # 카메라 실행
if cap.isOpened() == False:  # 카메라 생성 확인
    exit()

while True:
    ret, img = cap.read()  # 현재 이미지 가져오기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # 얼굴 인식

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # 얼굴 부분만 가져오기

        id_, conf = recognizer.predict(roi_gray)  # 얼마나 유사한지 확인
        print(id_, conf)  #2 52.171728 ..
        print(labelNames[id_])   #['boyoung', 'minho','minjae', txt]
        connection = pymysql.connect(user="root", password="1234", host="localhost", charset="utf8mb4")

        try:
            with connection.cursor() as cursor:
                sql = 'CREATE DATABASE cafe'
                sql = 'USE cafe'
                cursor.execute(sql)
                # sql = 'CREATE TABLE user (name varchar(10)) CHARSET utf8'
                # cursor.execute(sql)
             #   sql = 'CREATE TABLE customer (name varchar(10))CHARSET utf8'
              #  cursor.execute(sql)
                #cursor.execute(sql)
                sql = 'INSERT INTO customer values(%s)'
                #cursor.execute(sql, ('boyoung'))
                #cursor.execute(sql, ('minho'))
                cursor.execute(sql, (labelNames[id_]))

                connection.commit()
                result2 = cursor.fetchall()
                print(result2)
                print(labelDics)
        finally:
                 connection.close()


        if (labelNames[id_] in labelDics):
            font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 지정
            name = labelNames[id_]  # ID를 이용하여 이름 가져오기
            cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else :
            cv2.putText(img, "unknown", (x, y), font, 1, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('Preview', img)  # 이미지 보여주기


    if cv2.waitKey(10) >= 0:  # 키 입력 대기, 10ms
        break

cap.release()
cv2.destroyAllWindows()

