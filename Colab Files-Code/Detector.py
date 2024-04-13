import cv2

classNames= []
classFile = 'classes.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('n').split('\n')

#Configuration Files
configPath = 'yolov4_test.cfg'
weightsPath = 'yolov4_train_final.weights'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img,thres,nms,draw=True):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    objects = classNames
    #print(bbox)
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=1)
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img,objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)
    
    #Input Image Path
    img = cv2.imread("outdoor-archery-targets-on-grass-260nw-442491304.jpg")
    
    result,objectInfo = getObjects(img,0.45,0.2)
    #print(objectInfo)
    cv2.imshow("Output", img)
    print("hello")
    cv2.waitKey()
    
    '''
    while True:
        success, img = cap.read()
        result,objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        cv2.imshow("Output", img)
        cv2.waitKey(1)
    '''