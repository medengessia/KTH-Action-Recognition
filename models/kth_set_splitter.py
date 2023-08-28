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
    









# For using yolo...

# res = yolo(img)
# if len(res.xyxy[0].tolist()) != 0:
    # box = res.xyxy[0].tolist()[0]
    # label_ = res.names[int(box[-1])]

    # former_y, former_x = img.shape[0], img.shape[1]
    # change = False
    # crop_params = {
    # 'xmin' : 0,
    # 'ymin' : 0,
    # 'xmax' : former_x,
    # 'ymax' : former_y
    # }

    # if label_ == 'person' and ((box[0] < box[2] and box[1] < box[3]) and (box[0] > 0 and box[1] > 0 and box[2] < img.shape[0] and box[3] < img.shape[1])):
        # xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        # crop_params['xmin'] = max(0, int(xmin))
        # crop_params['ymin'] = max(0, int(ymin))
        # crop_params['xmax'] = min(former_x, int(xmax))
        # crop_params['ymax'] = min(former_y, int(ymax))
        # change = True

    # if change:
        # img = img[crop_params['ymin']:crop_params['ymax'], crop_params['xmin']:crop_params['xmax'], :]

    # img = cv2.resize(img, (former_x,former_y))













#frames_to_wipe_out = []
#for i in range(len(self.data)):
#    for j in range(len(self.data[i][0])-1):
#        img = cv2.imread(self.data[i][0][j+1]) - cv2.imread(self.data[i][0][j])
#        if img.all() < 1e-5:
#           frames_to_wipe_out.append(self.data[i][0][j])
#    for j in range(len(frames_to_wipe_out)):
#        if frames_to_wipe_out[j] in self.data[i][0]:
#           self.data[i][0].remove(frames_to_wipe_out[j])