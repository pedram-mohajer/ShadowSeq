import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import random

class GTSRBSequenceDataset:
    def __init__(self, csv_path, image_folder, sequences=None):
        """
        If sequences are provided (pre-split), load them directly.
        Otherwise, build them from the CSV.
        """
        self.image_folder = image_folder
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])
        self.sequences = []

        if sequences is not None:
            self.sequences = sequences
            return

        self.data = pd.read_csv(csv_path)
        grouped = self.data.groupby("SequenceID")

        for seq_id, group in tqdm(grouped, desc="Loading Sequences of Images"):
            if len(group["ClassId"].unique()) > 1:
                print(f"Warning: SequenceID={seq_id} has multiple ClassIds. Using first one only.")

            seq_dict = {
                'tensors': [],
                'rgb': [],
                'filenames': [],
                'sequenceId': seq_id,
                'classId': group["ClassId"].iloc[0]
            }

            for row in group.itertuples():
                image_path = os.path.join(self.image_folder, row.Filename)
                if not os.path.exists(image_path):
                    continue

                pil_img = Image.open(image_path).convert("RGB")
                cropped_img = pil_img.crop((row.Roi_X1, row.Roi_Y1, row.Roi_X2, row.Roi_Y2))
                resized_img = cropped_img.resize((128, 128), Image.BILINEAR)

                rgb_array = np.array(resized_img)
                img_tensor = self.transform(resized_img)

                seq_dict['tensors'].append(img_tensor)
                seq_dict['rgb'].append(rgb_array)
                seq_dict['filenames'].append(row.Filename)

            if seq_dict['tensors']:
                self.sequences.append(seq_dict)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def split_train_test(self, train_count=805, seed=42):
        """
        Splits the dataset into two GTSRBSequenceDataset objects (train, test)
        based on sequence-level shuffling.
        """
        random.seed(seed)
        shuffled = self.sequences.copy()
        random.shuffle(shuffled)

        train_seqs = shuffled[:train_count]
        test_seqs = shuffled[train_count:]

        train_set = GTSRBSequenceDataset(csv_path=None, image_folder=self.image_folder, sequences=train_seqs)
        test_set  = GTSRBSequenceDataset(csv_path=None, image_folder=self.image_folder, sequences=test_seqs)
        return train_set, test_set
