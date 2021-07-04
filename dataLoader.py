import csv
import glob
import torch
import os

from PIL import Image
from torchvision.transforms import *
from collections import namedtuple

ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'label', 'path'])
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']

def default_loader(path):
    return Image.open(path).convert('RGB')

class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):

        with open(csv_file_labels) as csv_label:
            classes_dct = {}
            csv_reader = [line.strip() for line in csv_label]
            data = list(csv_reader)
            for i, item in enumerate(data):
                classes_dct[item] = i
                classes_dct[i] = item

        csv_data_ = []
        with open(csv_file_input) as csvin:
            csv_reader = csv.reader(csvin, delimiter=';')
            for row in csv_reader:
                item = ListDataJpeg(row[0],
                                    row[1],
                                    os.path.join(root, row[0])
                                    )
                if row[1] in classes_dct:
                    csv_data_.append(item)
        self.csv_data = csv_data_

        self.transform = transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
        ])

        self.classes_dict = classes_dct
        self.root = root
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)
        imgs = []
        for img_path in img_paths:          #Simplify
            img = self.loader(img_path)   # Data loader can go there
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        target_idx = self.classes_dict[item.label]

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)

        return (data, target_idx)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        #set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset
            diff = (num_frames - num_frames_necessary)
        frame_names = frame_names[offset:num_frames_necessary +
                                  offset:self.step_size]
        return frame_names


test_ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'path'])

class TestVideoFolder(torch.utils.data.Dataset):


    def __init__(self, root, csv_file_input, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):

        csv_data_ = []
        with open(csv_file_input) as csvin:
            csv_reader = csv.reader(csvin, delimiter=';')
            for row in csv_reader:
                #print(row)
                item = test_ListDataJpeg(row[0],
                                    os.path.join(root, row[0])
                                    )
                csv_data_.append(item)

        self.csv_data = csv_data_


        self.transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
        ])

        self.root = root
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)
        imgs = []
        for img_path in img_paths:          #Simplify
            img = self.loader(img_path)   # Data loader can go there
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)


        return (data, index)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        #set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
        frame_names = frame_names[offset:num_frames_necessary +
                                  offset:self.step_size]
        return frame_names
