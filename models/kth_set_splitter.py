import os
import cv2 as cv
import torch
import numpy as np
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F 


class KTHDataset(Dataset):
    def __init__(self, annotations_file, data_dir, subjects, seq_length, image_shape, subject_left=None, create_frames=False, transform=None):

        # Constructor
        self.paths = []
        self.data = []

        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.subjects = subjects
        self.seq_length = seq_length
        self.image_shape = image_shape
        self.transform = transform

        # When a certain subject has to be removed (LOSO)
        self.left = []  

        if subject_left:
            self.left = subject_left
        
        actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        self.label_encoder = dict(zip(actions, range(len(actions))))

        # When frames have not been created yet from videos
        if create_frames:
            self.paths = self.video_to_frames()

        self.data = self.load_sequences()


    def file_processing(self, to_frames):

        with open(self.annotations_file) as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = str(lines[i]).strip('\n')
        
        while '' in lines:
            lines.remove('')

        for line in tqdm(lines, desc='loading data....'):
            vid_path, rest = line.split('\t\t')
            subject, action, scenario = vid_path.split('_')
            annot = rest.split('\t')[1]
            action_segments = annot.split(', ')
    
            # from video to frames

            if to_frames:
                self.paths.append(vid_path)
                path = os.path.join(self.data_dir, 'videos/' + subject + '/' + vid_path + '_uncomp.avi')
                reader = cv.VideoCapture(path)
                output_directory = os.path.join(self.data_dir, 'frames/' + subject + '/' + vid_path)
                os.makedirs(output_directory, exist_ok=True)
                frame_count = 0
                while True:
                    ret, frame = reader.read()
                    if not ret:
                        break
                    frame_filename = os.path.join(output_directory, 'frame_' + f'{frame_count}'.rjust(3, '0') + '.jpg')
                    cv.imwrite(frame_filename, frame)
                    frame_count += 1
                reader.release()
                
            # data loading

            else:
                if subject not in self.subjects or subject in self.left:
                    continue
                path = os.path.join(self.data_dir, 'frames/' + subject + '/' + vid_path)

                frames = glob.glob(path+'/*.jpg')
                frames.sort()

                for segment in action_segments:
                    start, end = segment.split('-')
                    start, end = int(start)-1, int(end)
                    sequence = frames[start:end]
                    self.data.append([sequence, self.label_encoder[action]])
                    

    def video_to_frames(self):
        to_frames = True
        self.file_processing(to_frames)
        return self.paths

                    
    def load_sequences(self):
        to_frames = False
        self.file_processing(to_frames)
        return self.data


    def __len__(self):
        return len(self.data)


    def padding_right(self, sequence, length):
        return F.pad(sequence, (0, 0, 0, 0, 0, 0, 0, length - sequence.shape[0]))
    
    
    def padding_left(self, sequence, length):
        return F.pad(sequence, (0, 0, 0, 0, 0, 0, length - sequence.shape[0], 0))
    

    def resize(self, sequence, new_size):
        return F.interpolate(sequence, size=new_size, mode='bilinear', align_corners=False)
    

    def __getitem__(self, idx):
        sequence_paths, label = self.data[idx]
        sequence = []

        for path in sequence_paths:
            img = cv.imread(path)   
            sequence.append(img)

        sequence = np.array(sequence)
        sequence = torch.from_numpy(sequence)
        sequence = sequence.permute(0, 3, 1, 2) / 255.0  # Normalize sequences numerically and adapt to PyTorch tensor format
        

        # Whenever there is data augmentation precised in the parameters

        if self.transform and torch.rand(1) > 0.7:
            sequence = self.transform(sequence)


        sequence = self.resize(sequence, self.image_shape)  # Equalize the dimensions of an image


        # Motion Guided Frame Selection Technique

        if sequence.shape[0] < self.seq_length:
            sequence = self.padding_right(sequence, self.seq_length)

        ## to choose the topk frames
        diff = torch.mean(sequence[1:] - sequence[:-1], dim=[1, 2, 3])
        diff = torch.cat((torch.zeros(1), diff)).abs()
        values, indices = torch.topk(diff, k=self.seq_length)
        best_indices, _ = torch.sort(indices)
        sequence = sequence[best_indices]

        return sequence, label
