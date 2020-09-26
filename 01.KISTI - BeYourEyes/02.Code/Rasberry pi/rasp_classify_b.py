#import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
print('Loading the Trained Model')
model = tf.keras.models.load_model('/tmp/test.h5')
labels = [
        '2percent', 'demisoda_apple', 'milkis', 'chilsung_cider', 'cocacola', 'toreta', 'powerade', 'pepsi', 'pokari', 'fanta_orange'
    ]
# start = time.time()

# def make_img():
#     cap = cv2.VideoCapture(0)

#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#     while (cap.isOpened):
#         ret, frame = cap.read()
    
#         if ret:
#             cv2.imshow("video",frame)
        
#             k = cv2.waitKey(33)
#             if k == 27: # esc
#                 break
#             elif k == 32: #spacebar
#                 img = cv2.resize(frame, dsize=(224,224))
#                 cv2.imwrite('/tmp/capture_img.png', img)
#                 break
#         else:
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()


def predict_img():
    image_size = (224, 224)    

    # 이미지 경로 혹은 이미지를 변수로 받던지?
    img = keras.preprocessing.image.load_img(
        '/tmp/capture_img.png', target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]

    # 결과 출력  
    print("This image is %s" % (labels[np.argmax(score)]))
    return labels[np.argmax(score)]

def voice_guidance(result):
    # result 는 labels[np.argmax(score)] 스트링값임 
    # 음성파일이름을 라벨 이름과 맞춰놓으면 된다.
    pygame.mixer.init()
    pygame.mixer.music.load('./voice/' + result + '.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    print('voice_guidance')


def main():
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 640 x 480
    count = 0
    predict_result = 'Searching'
    voice_guidance('intro1')
    while (cap.isOpened):
        ret, frame = cap.read()    
        if ret:                    
            k = cv2.waitKey(33)
            if k == 27: # esc
                break
            elif k == 32: #spacebar
                img = cv2.resize(frame, dsize=(224,224))
                cv2.imwrite('/tmp/capture_img.png', img)
                predict_result = predict_img()
                voice_guidance(predict_result)
            cv2.putText(frame, predict_result,(5,55),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), thickness=4)
            cv2.imshow("video",frame)            
            # 일정시간 지나고 화면에 글씨 다시 Searching으로 바꿔주기
            if count == 100:
                predict_result = 'Searching'
                count = 0
            elif predict_result != 'Searching':
                count += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
main()