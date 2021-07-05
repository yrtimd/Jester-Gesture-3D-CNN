import csv
import cv2
import pandas as pd
import glob
from random import randint
import os
import numpy
from torchvision.transforms import *
from PIL import Image
import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']


def playTrainVideo(path,fps = 5, ifTest=False):

    if ifTest:
    	train_vidoes_csv = pd.read_csv("./trainings/3D_CNN_models/cs523_project_model/jester-test-results.csv", header=None, sep = ";")
    	train_vidoes_csv = pd.DataFrame(train_vidoes_csv)
    	train_vidoes_csv[0] = train_vidoes_csv[0].astype(str) + train_vidoes_csv[1]
    else:
    	train_vidoes_csv = pd.read_csv(path, header=None)
    	train_vidoes_csv = pd.DataFrame(train_vidoes_csv)
    	

    video_folder = './20bn-jester-v1/videos'

    train_videos_split = train_vidoes_csv[0].str.split(";", expand=True)
    train_videos_split = train_videos_split.to_records(index=False)


    for i in range(3):
        value = randint(1, len(train_videos_split))
        window_name = str(train_videos_split[value][1])
        gesture_id = str(train_videos_split[value][0])

        ##Window name
        cv2.namedWindow(window_name)

        #get frame names
        frames_names = get_frames(os.path.join(video_folder, gesture_id))

        for path in frames_names:
                frame = cv2.imread(str(path))
                frame = cv2.resize(frame, (400, 400))
                cv2.imshow(window_name, frame)
                
                if (cv2.waitKey(int(1000/fps)) == 27):
                    break

        cv2.destroyWindow(window_name)


def videoPrediction(classes_dct, model, gesture_id):
    transform = Compose([
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
            ])

    video_folder = './20bn-jester-v1/videos'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = 3
    
    frames_names = get_frames(os.path.join(video_folder, str(gesture_id)))

    imgs = []
    for index, path in enumerate(frames_names):
        img = Image.open(path)
        img = transform(img)
        imgs.append(torch.unsqueeze(img, 0))
        #if index > 16:
        #    break

    data = torch.cat(imgs)
    data = data.permute(1, 0, 2, 3)
    data = torch.unsqueeze(data, 0)
	
    with torch.no_grad():
    	input= data.to(device)

    	# compute output and loss
    	output = model(input)

    	_, predicted = torch.max(output.data, 1)
    	predicted = predicted.detach().cpu().numpy()

    #print(window_name)
    window_name = classes_dct[int(predicted)]

    #print(predicted)
    cv2.namedWindow(window_name)
    
    for path in frames_names:          
        frame = cv2.imread(str(path))
        frame = cv2.resize(frame, (400, 400))
        cv2.imshow(window_name, frame)
        
        if (cv2.waitKey(int(1000/fps)) == 27):
            break
            
    cv2.destroyWindow(window_name)
    torch.cuda.empty_cache()




def play_video(classes_dct, model, video, fps = 20):

    transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs = []
    counter = 0
    vid = cv2.VideoCapture(0)
    
    if not vid.isOpened():
        print("Cannot open camera")
        exit()
    
    cv2.namedWindow('Video')
    ret_val = True
    while ret_val:
        ret_val, frame = vid.read()
        if not ret_val:
            print('frame broke')
            continue
        frame = cv2.resize(frame, (400, 400))
        cv2.imshow('Video', frame)
        
        
        if (cv2.waitKey(int(1000/fps)) == 27):
            break
        
        imgFrame = Image.fromarray(frame)
        imgFrame = transform(imgFrame)
        imgs.append(torch.unsqueeze(imgFrame, 0))
        counter += 1
        if counter > 17:
            data = torch.cat(imgs)
            data = data.permute(1, 0, 2, 3)
            data = torch.unsqueeze(data, 0)
            
            with torch.no_grad():
            	input= data.to(device)

            	# compute output and loss
            	output = model(input)

            	_, predicted = torch.max(output.data, 1)
            	predicted = predicted.detach().cpu().numpy()

            #print(window_name)
            print(classes_dct[int(predicted)])
            imgs = []
            counter = 0
            print('Singal: Do An Action')
            
    cv2.destroyWindow('Video')
    vid.release()
    torch.cuda.empty_cache()





def get_frames(path):
    frame_names = []
    for ext in IMG_EXTENSIONS:
        frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
    frame_names = list(sorted(frame_names))
    num_frames = len(frame_names)

    #set number of necessary frames
    num_frames_necessary = 36


    # pick frames
    offset = 0
    if num_frames_necessary > num_frames:
        # pad last frame if video is shorter than necessary
        frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
    frame_names = frame_names[offset:num_frames_necessary +
                              offset:2]
    return frame_names
