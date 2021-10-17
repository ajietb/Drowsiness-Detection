import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from pygame import mixer

plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
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

size = 100
x_vec = np.linspace(0,1,size+1)[0:-1]
y_vec = np.random.randn(len(x_vec))*0
line1 = []
record_path = "/home/ajietb/Supir ngantuk/dMEAR.csv"
raw_data = []
counter = 0
counter2 = 0
kedip = 0
kedip_value = 0
kantuk_state = False

if not os.path.isfile(record_path):
  print("Create New Data MEAR")
  record_df = pd.DataFrame(columns=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13", "Label"])
else:
  print("Open Existing Data MEAR")
  record_df = pd.read_csv(record_path)

model_path = "/home/ajietb/Supir ngantuk/TSmodel.h5"
if os.path.exists(model_path):
  print("Load Model From ", model_path)
  model = tf.keras.models.load_model(model_path)
else:
  print("Model  Doesn't Exist")
  while True:
    pass

mixer.init()
sound = mixer.Sound('/home/ajietb/Supir ngantuk/drowsiness/alarm.wav')
#======================================================================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
index = [246, 7, 161, 163, 160, 144, 159, 145, 158, 153, 157, 154, 173, 155, 33, 133, #Left Eye
        398, 382, 384, 381, 385, 380, 386, 374, 387, 373, 388, 390, 466, 249, 362, 263, #Right Eye
        78, 191, 95, 80, 88, 81, 178, 82, 87, 13, 14, 312, 317, 311, 402, 310, 318, 415, 324, 308] #Mouth 

d_index = {}
left_ratio = []
right_ratio = []
total_time = 0
previous_time = time.time()

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  while cap.isOpened():
    
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
        for id,lm in enumerate(face_landmarks.landmark):
            ih, iw, ic = image.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            if id in index:
              d_index["{}".format(id)] = [x,y]
              # cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
              #          0.7, (0, 255, 0), 1)
    
    for l in range(8):
      try:
        left_ratio.append(dist.euclidean(d_index["{}".format(index[2*l])], d_index["{}".format(index[2*l+1])]))
        right_ratio.append(dist.euclidean(d_index["{}".format(index[(2*l)+16])], d_index["{}".format(index[(2*l+1)+16])]))
      except:
        continue

    EAR_L = np.sum(left_ratio[0:-1])/(2*left_ratio[-1])
    EAR_R = np.sum(right_ratio[0:-1])/(2*right_ratio[-1])

    MEAR = (EAR_L+EAR_R)/2

    # print(MEAR)
    
    left_ratio = []
    right_ratio = []

    delta_time = time.time() - previous_time
    previous_time = time.time()

    total_time += delta_time
    #==============================
    y_vec[-1] = MEAR
    line1 = live_plotter(x_vec,y_vec,line1,identifier="MEAR", pause_time=0.00001)
    y_vec = np.append(y_vec[1:],0.0)
    #==============================
    if len(raw_data)==13:
      # new_row = {"F1":raw_data[0],
      #           "F2":raw_data[1],
      #           "F3":raw_data[2],
      #           "F4":raw_data[3],
      #           "F5":raw_data[4],
      #           "F6":raw_data[5],
      #           "F7":raw_data[6],
      #           "F8":raw_data[7],
      #           "F9":raw_data[8],
      #           "F10":raw_data[9],
      #           "F11":raw_data[10],
      #           "F12":raw_data[11],
      #           "F13":raw_data[12], 
      #           "Label":"-"}
      # record_df = record_df.append(new_row, ignore_index=True)
      # record_df.to_csv(record_path, index=0)
      # print("Saving Data to {}".format(record_path))

      #============================
      data = np.expand_dims(np.array(raw_data), 0)
      data_predict = model.predict(data)
      print(np.argmax(data_predict))
      
      if np.argmax(data_predict) == 2:
        counter += 1
        if counter >5:
          try:
            sound.play()
          except:
            pass
      else:
        counter = 0
        sound.stop()
      
      if np.argmax(data_predict)==1 and not kantuk_state:
        kantuk_state = True
      
      if kantuk_state:
        counter2 += 1
        kedip_value += np.argmax(data_predict)
      
      if counter2>15:
        if kedip_value>=9:
          kedip += 1 
          counter2 = 0
          kedip_value = 0
          kantuk_state = False
      
      #============================
      raw_data.pop(0)
      raw_data.append(MEAR)
      

    else:
      raw_data.append(MEAR)

    #==============================

    cv2.putText(image, f'FPS: {int(1/delta_time)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
    cv2.putText(image, f'Kedip: {int(kedip)}', (20, 120), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()