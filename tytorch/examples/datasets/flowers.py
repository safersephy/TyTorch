from pathlib import Path
import torch
import numpy as np
from tytorch.data import DatasetFactory, TyTorchDataset
from tytorch.data_utils import iter_valid_paths,load_image,check_create_folder
from dataclasses import dataclass
from torchvision import transforms
from PIL import Image

@dataclass
class FactorySettings():
    source_url: str
    bronze_folder: Path
    bronze_filename: str
    silver_folder: Path
    silver_filename:str
    unzip:bool
    formats:list
    valid_frac:float
    test_frac:float
    
@dataclass
class ImgFactorySettings(FactorySettings):
    image_size: tuple
    
    
    


class FlowersDatasetFactory(DatasetFactory):
    def __init__(
        self,       
        settings: ImgFactorySettings
        ):
        super().__init__(
            )
        self.settings = settings


    def transform(
        self, 
    ):

        paths_, class_names = iter_valid_paths(
            self.settings.bronze_folder / "flower_photos", formats=self.settings.formats
        )
  


        all_data = {
            "data": [],
            "labels": [],
        }
        for path in paths_:
            img = load_image(path, self.settings.image_size)
            x_ = np.transpose(img, (2, 0, 1))
            x = torch.tensor(x_ / 255.0).type(torch.float32)  # type: ignore
            y = torch.tensor(class_names.index(path.parent.name))
            all_data["data"].append(x)
            all_data["labels"].append(y)

        all_data["data"] = torch.stack(all_data["data"])  # Stack to create a tensor
        all_data["labels"] = torch.tensor(all_data["labels"])

        num_all_samples = len(all_data["data"])
        all_indices = torch.randperm(num_all_samples)
        
        test_size = int(num_all_samples * self.settings.test_frac)
        train_size = num_all_samples - test_size

        train_indices = all_indices[:train_size]
        test_indices = all_indices[train_size:]
        
        traintest_data_split = all_data["data"][train_indices]
        traintest_labels_split = all_data["labels"][train_indices]
        test_data_split = all_data["data"][test_indices]
        test_labels_split = all_data["labels"][test_indices]
        
        num_train_samples = len(traintest_data_split)
        split_indices = torch.randperm(num_train_samples) 
 
        val_size = int(num_train_samples * self.settings.valid_frac)
        train_size = num_train_samples - val_size       
               
        train_indices = split_indices[:train_size]
        val_indices = split_indices[train_size:]

        train_data_split = traintest_data_split[train_indices]
        train_labels_split = traintest_labels_split[train_indices]
        val_data_split = traintest_data_split[val_indices]
        val_labels_split = traintest_labels_split[val_indices]            
        
        # # Define your augmentation transformations
        # augmentation_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])              

        # augmented_train_data = []
        # augmented_train_labels = []

        # # Apply augmentation to training data only
        # for i in range(len(train_data_split)):
        #     img = train_data_split[i]
        #     label = train_labels_split[i]

        #     augmented_train_data.append(img)
        #     augmented_train_labels.append(label)

        #     img = transforms.ToPILImage()(img)

        #     for i in range(2):

        #         # Apply transformations
                
        #         augmented_img = augmentation_transform(img)
        #         #x_augmented = augmented_img  / 255.0  # Normalize manually
        #         x_augmented_tensor = torch.tensor(augmented_img).type(torch.float32) # Create tensor from the original image

        #         augmented_train_data.append(x_augmented_tensor)
        #         augmented_train_labels.append(label)

        # # Stack the augmented data and convert back to tensors
        # augmented_train_data = torch.stack(augmented_train_data)
        # augmented_train_labels = torch.tensor(augmented_train_labels)

        data = {
            "traindata": train_data_split,
            "trainlabels": train_labels_split,
            "validdata": val_data_split,
            "validlabels": val_labels_split,
            "testdata": test_data_split,
            "testlabels": test_labels_split,
        }

        check_create_folder(self.settings.silver_folder)
        torch.save(data, self.settings.silver_folder / self.settings.silver_filename)  
         
    # def transform(
    #     self, 
    # ):

    #     paths_, class_names = iter_valid_paths(
    #         self.settings.bronze_folder / "flower_photos", formats=self.settings.formats
    #     )

    #     # Create data structure for splitting
    #     all_data = {
    #         "data": [],
    #         "labels": [],
    #     }

    #     for path in paths_:
    #         img = load_image(path, self.settings.image_size)
    #         x_ = np.transpose(img, (2, 0, 1))
    #         x = torch.tensor(x_ / 255.0).type(torch.float32)  # Normalize manually
    #         y = torch.tensor(class_names.index(path.parent.name))
    #         all_data["data"].append(x)
    #         all_data["labels"].append(y)

    #     # Convert lists to tensors
    #     all_data["data"] = torch.stack(all_data["data"])  # Stack to create a tensor
    #     all_data["labels"] = torch.tensor(all_data["labels"])

    #     # Split the dataset before augmentation
    #     num_all_samples = len(all_data["data"])
    #     all_indices = torch.randperm(num_all_samples)
        
    #     test_size = int(num_all_samples * self.settings.test_frac)
    #     train_size = num_all_samples - test_size

    #     train_indices = all_indices[:train_size]
    #     test_indices = all_indices[train_size:]

    #     train_data_split = all_data["data"][train_indices]
    #     train_labels_split = all_data["labels"][train_indices]
    #     test_data_split = all_data["data"][test_indices]
    #     test_labels_split = all_data["labels"][test_indices]

    #     # Further split training data into training and validation sets
    #     num_train_samples = len(train_data_split)
    #     split_indices = torch.randperm(num_train_samples) 

    #     val_size = int(num_train_samples * self.settings.valid_frac)
    #     train_size = num_train_samples - val_size       

    #     train_indices = split_indices[:train_size]
    #     val_indices = split_indices[train_size:]

    #     val_data_split = train_data_split[val_indices]
    #     val_labels_split = train_labels_split[val_indices]
    #     train_data_split = train_data_split[train_indices]
    #     train_labels_split = train_labels_split[train_indices]

    #     # Define your augmentation transformations (apply only to the training set)
    #     augmentation_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])      

    #     augmented_train_data = []
    #     augmented_train_labels = []

    #     # Apply augmentation to training data only
    #     for i in range(len(train_data_split)):
    #         img = train_data_split[i]
    #         label = train_labels_split[i]
            
    #         # Augment multiple times for each image if needed
    #         for _ in range(5):
    #             img_augmented = augmentation_transform(img.permute(1, 2, 0))  # Unpermute for PIL compatibility
    #             augmented_train_data.append(img_augmented)
    #             augmented_train_labels.append(label)

    #     # Stack the augmented data and convert back to tensors
    #     augmented_train_data = torch.stack(augmented_train_data)
    #     augmented_train_labels = torch.tensor(augmented_train_labels)

    #     # Save splits
    #     data = {
    #         "traindata": augmented_train_data,
    #         "trainlabels": augmented_train_labels,
    #         "validdata": val_data_split,
    #         "validlabels": val_labels_split,
    #         "testdata": test_data_split,
    #         "testlabels": test_labels_split,
    #     }

    #     check_create_folder(self.settings.silver_folder)
    #     torch.save(data, self.settings.silver_folder / self.settings.silver_filename)
        
    def load(
        self,
    ) -> tuple[
        TyTorchDataset,
        TyTorchDataset,
        TyTorchDataset,
    ]:
        data = torch.load(self.settings.silver_folder / self.settings.silver_filename, weights_only=False)  # type: ignore

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
            self.settings.source_url,
            self.settings.bronze_folder,
            self.settings.bronze_filename,
            self.settings.unzip
        )
        self.transform()
        return self.load()


flowers_factory_settings  = ImgFactorySettings(
    source_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    bronze_folder = Path("./tytorch/examples/data/bronze"), 
    bronze_filename="flowers.tgz",
    silver_folder = Path("./tytorch/examples/data/silver"),
    silver_filename="flowers.pt", 
    valid_frac=.2,
    test_frac=.2,
    unzip=True,
    formats=['.jpg','.png'],
    image_size=(224, 224)
)

# train_dataset, valid_dataset, test_dataset = FlowersDatasetFactory(flowers_factory_settings
# ).create_datasets()