from pathlib import Path
import torch
import numpy as np
from tytorch.data import DatasetFactory, TyTorchDataset
from tytorch.data_utils import iter_valid_paths,load_image,check_create_folder

class FlowersDatasetFactory(DatasetFactory):
    def __init__(
        self, 
        bronze_folder = None, 
        bronze_filename = None,        
        silver_folder = None, 
        test_fraction = None, 
        val_fraction = None,
        silver_filename = None,
        image_size = None
        ):
        super().__init__(
            bronze_folder, 
            bronze_filename
            )

        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.image_size = image_size
        self.silver_folder=silver_folder
        self.silver_filename=silver_filename

    def transform(
        self, 
    ):

        formats = ['.jpg','.png']
        paths_, class_names = iter_valid_paths(
            self.bronze_folder / "flower_photos", formats=formats
        )
        

        all_data = {
            "data": [],
            "labels": [],
        }
        for path in paths_:
            img = load_image(path, self.image_size)
            x_ = np.transpose(img, (2, 0, 1))
            x = torch.tensor(x_ / 255.0).type(torch.float32)  # type: ignore
            y = torch.tensor(class_names.index(path.parent.name))
            all_data["data"].append(x)
            all_data["labels"].append(y)

        all_data["data"] = torch.stack(all_data["data"])  # Stack to create a tensor
        all_data["labels"] = torch.tensor(all_data["labels"])

        num_all_samples = len(all_data["data"])
        all_indices = torch.randperm(num_all_samples)
        
        test_size = int(num_all_samples * self.test_fraction)
        train_size = num_all_samples - test_size

        train_indices = all_indices[:train_size]
        test_indices = all_indices[train_size:]
        
        traintest_data_split = all_data["data"][train_indices]
        traintest_labels_split = all_data["labels"][train_indices]
        test_data_split = all_data["data"][test_indices]
        test_labels_split = all_data["labels"][test_indices]
        
        num_train_samples = len(traintest_data_split)
        split_indices = torch.randperm(num_train_samples) 
 
        val_size = int(num_train_samples * self.val_fraction)
        train_size = num_train_samples - val_size       
               
        train_indices = split_indices[:train_size]
        val_indices = split_indices[train_size:]

        train_data_split = traintest_data_split[train_indices]
        train_labels_split = traintest_labels_split[train_indices]
        val_data_split = traintest_data_split[val_indices]
        val_labels_split = traintest_labels_split[val_indices]            
        

        data = {
            "traindata": train_data_split,
            "trainlabels": train_labels_split,
            "validdata": val_data_split,
            "validlabels": val_labels_split,
            "testdata": test_data_split,
            "testlabels": test_labels_split,
        }

        check_create_folder(self.silver_folder)
        torch.save(data, self.silver_folder / self.silver_filename)  
         

    def load(
        self,
    ) -> tuple[
        TyTorchDataset,
        TyTorchDataset,
        TyTorchDataset,
    ]:
        data = torch.load(self.silver_folder / self.silver_filename, weights_only=False)  # type: ignore

        train_data = data["traindata"]
        train_labels = data["trainlabels"]
        valid_data = data["validdata"]
        valid_labels = data["validlabels"]
        test_data = data["testdata"]
        test_labels = data["testlabels"]

        train_dataset = TyTorchDataset(train_data, train_labels)
        valid_dataset = TyTorchDataset(valid_data, valid_labels)
        test_dataset = TyTorchDataset(test_data, test_labels)

        return train_dataset, valid_dataset, test_dataset

    def create_datasets(self):
        self.extract(
            source_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            bronze_filename="flowers.tgz",
            unzip=True,
        )
        self.transform(silver_filename="flowers.pt", image_size=(224, 224))
        return self.load()


train_dataset, valid_dataset, test_dataset = FlowersDatasetFactory(
   silver_folder = Path("./tytorch/examples/data/silver"),
   silver_filename = "flowers.pt"
).load()