from torch.utils.data.dataset import Dataset


class skeletonDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        """
        Create new object of the dataset.
        Assuming the data_folder is in the following structure:
        |--- data_folder
            |--- Sub001
                |--- Left
                    |--- Back
                        |--- depth_frames
                        |--- rgb_frames
                        |--- log.json
                        |--- Sub001_Left_Back.csv
                    |--- Front
                        |--- ...
                    |--- Side
                        |--- ...
                |--- Right
                |--- Squat
                |--- Stand
                |--- Tight
            |--- Sub002
            |--- Sub003
            ...

        :param data_folder: Path to where the data is.
        :param transform: torchvision.transforms object.
        """
        self.data_folder = data_folder
        self.transform = transform

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.labels)
