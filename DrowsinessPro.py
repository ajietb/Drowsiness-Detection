from typing import SupportsComplex
from scipy.spatial import distance as dist
from pygame import mixer

import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import cv2
import time
import os

plt.style.use('ggplot')

class drowsiness:
    def __init__(self, n_face = 1, refine = True, min_detect_conf = 0.5, min_track_conf = 0.5, detect = False):
        plt.style.use('ggplot')
        mixer.init()

        self.detect = detect
        
        self.size = 100
        self.x_vec = np.linspace(0,1,self.size+1)[0:-1]
        self.y_vec = np.random.randn(len(self.x_vec))*0
        self.line1 = []
        self.raw_data = []
        self.d_index = {}
        self.left_ratio = []
        self.right_ratio = []
        self.counter = 0
        self.counter2 = 0
        self.kedip = 0
        self.kedip_value = 0
        self.total_time = 0
        self.kedip_state = False
        self.get_data = False
        self.sound_state = False
        self.previous_time = time.time()

        self.record_path = "/home/ajietb/Supir ngantuk/drowsiness/dMEAR.csv"
        self.sound_path = "/home/ajietb/Supir ngantuk/drowsiness/alarm.wav"
        self.model_path = "/home/ajietb/Supir ngantuk/drowsiness/TSmodel (1).h5"

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.index = [246, 7, 161, 163, 160, 144, 159, 145, 158, 153, 157, 154, 173, 155, 33, 133, #Left Eye
                    398, 382, 384, 381, 385, 380, 386, 374, 387, 373, 388, 390, 466, 249, 362, 263, #Right Eye
                    78, 191, 95, 80, 88, 81, 178, 82, 87, 13, 14, 312, 317, 311, 402, 310, 318, 415, 324, 308] #Mouth 
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=n_face,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=min_detect_conf,
                                                    min_tracking_confidence=min_track_conf)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Video Capture Fail, Try Another Index Camera")
        else:
            print("Camera Opened Successfully")

        if not os.path.isfile(self.record_path):
            print("Create New Data MEAR")
            self.record_df = pd.DataFrame(columns=["F1","F2","F3","F4","F5","F6","F7", "Label"])#,"F8","F9","F10","F11","F12","F13"
        else:
            print("Open Existing Data MEAR")
            self.record_df = pd.read_csv(self.record_path)
        
        if os.path.exists(self.model_path):
            print("Load Model From ", self.model_path)
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            raise ValueError("Model  Doesn't Exist")
        
        self.sound = mixer.Sound(self.sound_path)
    
    def live_plotter(self, x_vec,y1_data,line1,identifier='',pause_time=0.1):
        if line1==[]:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
            #update plot label/title
            plt.ylabel('Y Label')
            plt.title('Title: {}'.format(identifier))
            plt.show()
        
        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        
        # return line so we can update it again in the next iteration
        return line1
    
    def access_camera(self):
        self.success, self.image = self.cap.read()
        self.image = cv2.flip(self.image, 1)

    def start(self):
        while self.cap.isOpened():
            self.success, self.image = self.cap.read()
            self.image = cv2.flip(self.image, 1)
            if not self.success:
                print("Ignoring empty camera frame.")
                continue
            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.results = self.face_mesh.process(self.image)
            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            if self.results.multi_face_landmarks:
                for face_landmarks in self.results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=self.image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=self.image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    self.mp_drawing.draw_landmarks(
                        image=self.image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                    
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = self.image.shape
                        x,y = int(lm.x*iw), int(lm.y*ih)
                        if id in self.index:
                            self.d_index["{}".format(id)] = [x,y]
                            # cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                            #          0.7, (0, 255, 0), 1)
            
            for l in range(8):
                try:
                    self.left_ratio.append(dist.euclidean(self.d_index["{}".format(self.index[2*l])], 
                                                        self.d_index["{}".format(self.index[2*l+1])]))
                    self.right_ratio.append(dist.euclidean(self.d_index["{}".format(self.index[(2*l)+16])], 
                                                        self.d_index["{}".format(self.index[(2*l+1)+16])]))
                except:
                    continue
            try:
                self.EAR_L = np.sum(self.left_ratio[0:-1])/(2*self.left_ratio[-1])
                self.EAR_R = np.sum(self.right_ratio[0:-1])/(2*self.right_ratio[-1])
                self.MEAR = (self.EAR_L+self.EAR_R)/2
                if self.MEAR>0.35:
                    self.MEAR = 0.6
                else:
                    self.MEAR = 0.1
            except:
                continue

            self.left_ratio = []
            self.right_ratio = []

            self.delta_time = time.time() - self.previous_time
            self.previous_time = time.time()
            self.total_time += self.delta_time

            #=================================================================
            #Live Plotter
            self.y_vec[-1] = self.MEAR
            self.line1 = self.live_plotter(self.x_vec,self.y_vec,self.line1,identifier="MEAR", pause_time=0.00001)
            self.y_vec = np.append(self.y_vec[1:],0.0)
            #=================================================================

            if len(self.raw_data) == 7:
                if self.get_data:
                    self.new_row = {"F1":self.raw_data[0],
                                    "F2":self.raw_data[1],
                                    "F3":self.raw_data[2],
                                    "F4":self.raw_data[3],
                                    "F5":self.raw_data[4],
                                    "F6":self.raw_data[5],
                                    "F7":self.raw_data[6],
                                    # "F8":self.raw_data[7],
                                    # "F9":self.raw_data[8],
                                    # "F10":self.raw_data[9],
                                    # "F11":self.raw_data[10],
                                    # "F12":self.raw_data[11],
                                    # "F13":self.raw_data[12], 
                                    "Label":"-"}
                    self.record_df = self.record_df.append(self.new_row, ignore_index=True)
                    self.record_df.to_csv(self.record_path, index=0)
                    print("Saving Data to {}".format(self.record_path))
                    self.raw_data.pop(0)
                    self.raw_data.append(self.MEAR)
                elif self.detect:
                    self.data = np.expand_dims(np.array(self.raw_data), 0)
                    self.data_predict = self.model.predict(self.data)
                    # print(np.argmax(self.data_predict))
                    
                    if np.argmax(self.data_predict) == 2:
                        self.counter += 1
                        if self.counter>5 and not self.sound_state:
                            try:
                                self.sound.play(1000000000)
                            except:
                                pass
                            self.sound_state = True
                    else:
                        self.sound_state = False
                        self.counter = 0
                        self.sound.stop()
                        
                    if np.argmax(self.data_predict)==1 and not self.kedip_state:
                        self.kedip_state = True
                    
                    if self.kedip_state:
                        self.counter2 += 1
                        self.kedip_value += np.argmax(self.data_predict)
                    
                    if self.counter2>7:
                        if self.kedip_value>=3:
                            self.kedip += 1 
                            self.counter2 = 0
                            self.kedip_value = 0
                            self.kedip_state = False
                    
                    self.raw_data.pop(0)
                    self.raw_data.append(self.MEAR)
                else:
                    print("Get Data Paused")
            else:
                self.raw_data.append(self.MEAR)
            
            cv2.putText(self.image, 
                        f'FPS: {int(1/self.delta_time)}', 
                        (20, 70), 
                        cv2.FONT_HERSHEY_PLAIN,
                        3, 
                        (0, 255, 0), 3)
            cv2.putText(self.image, 
                        f'Kedip: {int(self.kedip)}', 
                        (20, 120), 
                        cv2.FONT_HERSHEY_PLAIN,
                        3, 
                        (0, 255, 0), 3)
            cv2.imshow('MediaPipe Face Mesh', self.image)
            
            self.wait = cv2.waitKey(1)

            if self.wait & 0xFF == ord('q'):
                break
            
            if self.wait == ord('o'):
                self.get_data = True
                self.detect = False
            elif self.wait == ord('p'):
                self.get_data = False
                self.detect = False
            elif self.wait == ord('d'):
                self.detect = True
                self.get_data = False
            else:
                pass

        self.cap.release()
                
if __name__ == "__main__":
    ds = drowsiness(n_face=1, 
                    refine = True, 
                    min_detect_conf = 0.5, 
                    min_track_conf = 0.5,
                    detect=False)
    ds.start()


    
