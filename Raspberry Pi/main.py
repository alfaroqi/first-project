#--- IMPORT DEPENDENCIES ------------------------------------------------------+
 
import os
import time
import requests
from datetime import datetime
# from firebase import firebase # but canceled using firebase
import smtplib
import json


import cv2
import numpy as np
import mysql.connector

# deep sort
from deep_sort.tracker import Tracker
from deep_sort import preprocessing 
from deep_sort import nn_matching
 
# utils
from utils import video_frame_gen
from utils import label_map
from utils import initialize_detector
from utils import initialize_img_source
from utils import generate_detections

# connection to database
connection = mysql.connector.connect(host='192.168.0.9',
                                        database='counting',
                                        user='root',
                                        password='admin')

mySql_insert_query = """INSERT INTO result (id, motor, mobil, bus_or_truk, waktu) 
                        VALUES (%s, %s, %s, %s, %s) """

cursor = connection.cursor()
current_Date = datetime.now()
# convert date in the format you want
formatted_date = current_Date.strftime('%Y-%m-%d %H:%M')


LABEL_PATH = "Model/labelmap.txt"
# MODEL_PATH = "Model/detect.tflite"
THRESHOLD = 0.5
DEFAULT_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), LABEL_PATH)
TRACKER_OUTPUT_TEXT_FILE = 'Laporan_'+(current_Date)+'.csv'
VIDEO_PATH = "sore.mp4"
CAMERA = False
DISPLAY = True
SAVE = False
tpu = False


# deep sort related
MAX_COSINE_DIST = 0.4
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0



# parse label map
labels = label_map(LABEL_PATH, DEFAULT_LABEL_MAP_PATH)

# initialize detector
interpreter = initialize_detector()

# create output directory
if not os.path.exists('output') and SAVE: os.makedirs('output')

# initialize deep sort tracker   
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
tracker = Tracker(metric) 
from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

t0 = time.time()
delta = 0
id = 0

bus = []
car = []
motorbike = []
truk = []



# initialize image source
img_generator = video_frame_gen(VIDEO_PATH)

# initialize plot colors (if necessary)
if DISPLAY or SAVE: COLORS = (np.random.rand(32, 3) * 255).astype(int)
duration = 5
# main tracking loop
print('\n> TRACKING...')
with open(TRACKER_OUTPUT_TEXT_FILE, 'w') as out_file:
    
    for i, pil_img in enumerate(img_generator):
    
       
        # print('> FRAME:', i)
        t1 = time.time()
        
        # add header to trajectory file
        if i == 0:
            header = (f'frame_num,time,obj_class,obj_id,obj_age,'
                'obj_t_since_last_update,obj_hits,'
                'xmin,ymin,xmax,ymax')
            print(header, file=out_file)

        # get detections
        detections = generate_detections(pil_img, interpreter, THRESHOLD)
        
        # proceed to updating state
        if len(detections) == 0: print('   > no detections...')
        else:
        
            # update tracker
            tracker.predict()
            tracker.update(detections)
            
            # save object locations
            if len(tracker.tracks) > 0:
                for track in tracker.tracks:
                    bbox = track.to_tlbr()
                    class_name = labels[track.get_class()]
                    row = (f'{i},{time.strftime("%c")},{class_name},'
                        f'{track.track_id},{int(track.age)},'
                        f'{int(track.time_since_update)},{str(track.hits)},'
                        f'{int(bbox[0])},{int(bbox[1])},'
                        f'{int(bbox[2])},{int(bbox[3])}')
                    print(row, file=out_file)
            
        # for get the last image output on the display
        if DISPLAY or SAVE:
        
            # convert pil image to cv2
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            current_motorbike = 0
            current_car= 0
            current_bus = 0
            current_truk = 0

            # cycle through actively tracked objects
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                # draw detections and label
                bbox = track.to_tlbr()
                class_name = labels[track.get_class()]
                color = COLORS[int(track.track_id) % len(COLORS)].tolist()
                cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(class_name))+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(cv2_img, str(class_name) + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                pts[track.track_id].append(center)


                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*2)
                    cv2.line(cv2_img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

                height, width, _ = cv2_img.shape
                cv2.line(cv2_img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
                cv2.line(cv2_img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)
                

                center_y = int(((bbox[1])+(bbox[3]))/2)
                
                

                if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
                    if class_name == 'bus' or class_name == 'truk': 
                        bus.append(int(track.track_id))
                        current_bus += 1
                    
                    elif class_name == 'mobil':
                        car.append(int(track.track_id))
                        current_car += 1
                        
                    elif class_name == 'motor':
                        motorbike.append(int(track.track_id))
                        current_motorbike += 1
    
    

                        
            total_car = len(set(car))
            total_bus = len(set(bus))
            # total_truk = len(set(truk))
            total_motor = len(set(motorbike))
            

            vehicleInArea = str(current_motorbike + current_car + current_truk + current_bus)
   
        
            cv2.putText(cv2_img, "Mobil: " + str(total_car), (0, 100), 0, 1, (0, 0, 255), 2)
            cv2.putText(cv2_img, "Motor: " + str(total_motor), (0,130), 0, 1, (0,0,255), 2)
            cv2.putText(cv2_img, "Bus / Truk: " + str(total_bus), (0,160), 0, 1, (0,0,255), 2)
            # cv2.putText(cv2_img, "motorDown: " + str(total_motorDown), (0,190), 0, 1, (0,0,255), 2)
            cv2.putText(cv2_img, "Kendaraan Terdeksi: " + str(len(detections)), (0,220), 0, 1, (0,0,255), 2)

            print('Mobil :',total_car, "Motor :",total_motor, "Bus or Truk  :",total_bus,  time.strftime("%c") )
    

            fps = 1./(time.time()-t1)
            cv2.putText(cv2_img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
            cv2.imshow("tracker output", cv2_img)
           
            k = cv2.waitKey(10) & 0xff
            delta += t1 - t0
            t0 = t1

        
            if delta > 60: # reset the timer
                id += 1
                
                #karena hanya sekitar 2-4 fps jadi video 1 detik dapat diproses selama 4.5 detik jika 1 menit jadi 270 detik 
                    
                    # cv2.imshow("tracker output", cv2_img)
                    # k = cv2.waitKey(10) & 0xff
                # if k == 27 :
                print("============================HASIL===============================")
                print("Mobil       : " + str(total_car))
                print("Motor       : " + str(total_motor))
                print("Bus / Truk  : " + str(total_bus))

                printCar = str(total_car)
                printMotor = str(total_motor)
                printBT = str(total_bus)
                
                # send data ke database
                insert_tuple = (id, printMotor, printCar, printBT, current_Date)
                result = cursor.execute(mySql_insert_query, insert_tuple)
                connection.commit()
                print("Date Record inserted successfully")
               
    
                
                bus = []
                car = []
                motorbike = []
                truk = []
                delta = 0
                
                
                
            if k & 0xFF == ord("q"): # quit all
                break
                        

    
    # last persist frames
    if SAVE: cv2.imwrite(f'output/frame_{i}.jpg', cv2_img)
      
cv2.destroyAllWindows()    
    
