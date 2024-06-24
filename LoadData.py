import json
import math
import os
import os.path
import random
import cv2
import numpy as np
import shutil

class NSLT:

    def __init__(self, split_file, split, root, mode, transforms=None, save_path_param=None):
        self.num_classes = self.get_num_class(split_file)
        
        self.data = self.make_dataset(split_file, split, root, mode, num_classes=self.num_classes, save_path=save_path_param)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.cwd = os.getcwd()


    def __len__(self):
        return len(self.data)



    def load_rgb_frames(image_dir, vid, start, num):
        frames = []
        for i in range(start, start + num):
            try:
                img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
            except:
                print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = (img / 255.) * 2 - 1
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)


    def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
        video_path = os.path.join(vid_root, vid + '.mp4')

        vidcap = cv2.VideoCapture(video_path)

        frames = []

        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(min(num, int(total_frames - start))):
            success, img = vidcap.read()

            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

            img = (img / 255.) * 2 - 1

            frames.append(img)

        return np.asarray(frames, dtype=np.float32)

    def make_dataset(self, split_file, split, root, mode, num_classes, save_path=None):
        print ('make dataset for ',split)
        
        dataset = []
        with open(split_file, 'r') as f:
            data = json.load(f)

        i = 0
        count_skipping = 0
        if (save_path==None):
            save_path = os.getcwd()
            print("save_path = ",save_path) 
        
        
        for vid in data.keys():
            # this code make the loop back to the main loop while does not meet the parameter such us: test, train, or val
            if split == 'train':
                if data[vid]['subset'] not in ['train', 'val']:
                    continue
            else:
                if data[vid]['subset'] != 'test':
                    continue
            #  take from the parameter which contain the dataset
            vid_root = root['word']
            src = 0

            video_path = os.path.join(vid_root, vid + '.mp4')

            # ## split to dataset and write in folder
            tempLabel = str (data[vid]['action'][0])
            if split == 'train':
                new_dir = str (os.path.join(save_path,'DataTraining100',tempLabel))
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                    # print(new_dir)
                shutil.copy(video_path, new_dir)
            elif split == 'val':
                new_dir = str (os.path.join(save_path,'DataValidation100',tempLabel))
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                    # print(new_dir)
                shutil.copy(video_path, new_dir)
            elif split == 'test':
                new_dir = str (os.path.join(save_path,'DataTesting100',tempLabel))
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                    # print(new_dir)
                shutil.copy(video_path, new_dir)
            
            i += 1
        # print("Skipped videos: ", count_skipping)
        # print(len(dataset))
        return dataset

    def get_num_class(self, split_file):
        classes = set()
        content = json.load(open(split_file))

        for vid in content.keys():
            class_id = content[vid]['action'][0]
            classes.add(class_id)

        return len(classes)

if __name__ == "__main__":
    cwd = os.getcwd()
    mode = 'rgb'
    root = {'word': os.path.join(cwd, 'WLASL2000')}
    train_split = os.path.join(cwd, 'preprocess/nslt_100.json')
    save_path_param = "/home/tamlab/Documents/SignLanguage/CodeInServer/Project2"
    
    
    # # ## copying dataset into folder
    # # ##next extract intokeypoint folder, firstly watch i3d method what is the effect on start and end of frame

    ## had been executed
    dataset         = NSLT(train_split, 'train', root, mode,None, save_path_param = save_path_param)
    dataset_test    = NSLT(train_split, 'test' , root, mode,None, save_path_param = save_path_param)
    dataset         = NSLT(train_split, 'val'  , root, mode,None, save_path_param = save_path_param)
