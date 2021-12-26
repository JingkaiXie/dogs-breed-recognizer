from PIL import Image
import csv
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_DIR = "dog breed data/images/Images"
breed_list = os.listdir(DATA_DIR)
LABEL_NAMES = ["-".join(b.split('-')[1:]) for b in breed_list]

FOLDERS = os.listdir(DATA_DIR)

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.img_names = []
        self.labels = []
        for f, b in zip(FOLDERS, LABEL_NAMES):
            file_names = os.listdir(DATA_DIR + '/' + f)
            for name in file_names:
                self.img_names.append(name)
                self.labels.append(LABEL_NAMES.index(b))  # can just use i = 0,1...n

        # with open(self.dataset_path + '/labels.csv', 'r') as csv_file:
        #     csv_reader = csv.reader(csv_file)
        #     next(csv_reader)
        #     for line in csv_reader:
        #         self.img_names.append(line[0])
        #         self.labels.append(LABEL_NAMES.index(line[1]))

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, idx):

        img = Image.open(self.dataset_path + '/' + self.img_names[idx]).convert('RGB')
        img_to_tensor = transforms.ToTensor()
        img_tensor = img_to_tensor(img)
        img_tensor = transforms.Resize(size=(200, 200))(img_tensor)

        return img_tensor, self.labels[idx]


def load_data(dataset_path, num_workers=0):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, shuffle=True, drop_last=True, batch_size=256)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
