import os
import numpy as np
import pandas as pd
from torchinfo import summary
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torchvision.transforms as T
import sys, torch, torchvision, torchmetrics, imageio

from tqdm import tqdm
from colorama import Fore
from sklearn import preprocessing
from skimage.transform import resize
from skimage import img_as_float32, img_as_ubyte
from sklearn.model_selection import train_test_split

# Minor tensor-core speedup
torch.set_float32_matmul_precision('medium')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Data Loader Class #
#########################################################################################


class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size = 64,
                 val_split = 0.2,
                 num_workers = 4,
                 location = './data',
                 download = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.location = location
        self.input_shape = None
        self.output_shape = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.download = download

        self.size = [64, 64]
        self.N = 27000
        self.extracted = '2750'
        self._load_data()

    def _load_data(self):
        images = np.zeros(
            [self.N, self.size[0], self.size[1], 3], dtype="uint8")
        labels = []
        filenames = []

        if self.download:
            self.download_dataset()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        i = 0
        data_dir = os.path.join(self.location, self.extracted)

        with tqdm(os.listdir(data_dir), bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)) as dir_bar:
            for item in dir_bar:
                f = os.path.join(data_dir, item)
                if os.path.isfile(f):
                    continue
                for subitem in os.listdir(f):
                    sub_f = os.path.join(f, subitem)
                    filenames.append(sub_f)

                    # a few images are a few pixels off, we will resize them
                    image = imageio.imread(sub_f)
                    if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
                        # print("Resizing image...")
                        image = img_as_ubyte(
                            resize(
                                image, (self.size[0], self.size[1]), anti_aliasing=True)
                        )
                    images[i] = img_as_ubyte(image)
                    i += 1
                    labels.append(item)

                dir_bar.set_postfix(category=item)

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # sort by filenames
        images = images[filenames.argsort()]
        labels = labels[filenames.argsort()]

        # convert to integer labels
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(np.sort(np.unique(labels)))
        labels = label_encoder.transform(labels)
        labels = np.asarray(labels)
        # remember label encoding
        self.label_encoding = list(label_encoder.classes_)

        self.data = images
        self.targets = labels
    
    def setup(self, stage: str):
        if (stage == 'fit' or \
            stage == 'validate') and \
            not(self.data_train and self.data_val):
            x_train = self.data.transpose((0,3,1,2))
            y_train = np.array(self.targets)
            self.input_shape = x_train.shape[1:]
            self.output_shape = (len(np.unique(y_train)),)
            rng = np.random.default_rng()
            permutation = rng.permutation(x_train.shape[0])
            split_point = int(x_train.shape[0]*(1.0-self.val_split))
            self.data_train = list(zip(torch.Tensor(x_train[permutation[:split_point]]).to(torch.float32),
                                       torch.Tensor(y_train[permutation[:split_point]]).to(torch.long)))
            self.data_val = list(zip(torch.Tensor(x_train[permutation[split_point:]]).to(torch.float32),
                                     torch.Tensor(y_train[permutation[split_point:]]).to(torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        image = np.asarray(img / 255, dtype="float32")

        return image.transpose(2, 0, 1), self.targets[idx]

    def _check_exists(self) -> bool:
        """
        Check the Root directory is exists
        """
        return os.path.exists(self.location)

    def download_dataset(self) -> None:
        """
        Download the dataset from the internet
        """

        if self._check_exists():
            return

        os.makedirs(self.location, exist_ok = True)
        torchvision.datasets.utils.download_and_extract_archive(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
            download_root = self.location,
            md5 = "c8fa014336c82ac7804f0398fcb19387",
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)


# Neural Network Classes #
#########################################################################################


class SinePositionEmbedding(pl.LightningModule):
    def __init__(self,
                 max_wavelength=10000.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = torch.Tensor([max_wavelength])

    def forward(self, x):
        input_shape = x.shape
        seq_length = x.shape[-2]
        hidden_size = x.shape[-1]
        position = torch.arange(seq_length).type_as(x)
        min_freq = (1 / self.max_wavelength).type_as(x)
        timescales = torch.pow(
            min_freq,
            (2 * (torch.arange(hidden_size) // 2)).type_as(x)
            / torch.Tensor([hidden_size]).type_as(x)
        )
        angles = torch.unsqueeze(position, 1) * torch.unsqueeze(timescales, 0)
        cos_mask = (torch.arange(hidden_size) % 2).type_as(x)
        sin_mask = 1 - cos_mask
        positional_encodings = (
            torch.sin(angles) * sin_mask + torch.cos(angles) * cos_mask
        )
        return torch.broadcast_to(positional_encodings, input_shape)

class MLP(pl.LightningModule):
    def __init__(self,
                 latent_size = 64,
                 dropout = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.linear1 = torch.nn.Linear(latent_size,
                                       latent_size)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(latent_size,
                                       latent_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(pl.LightningModule):
    def __init__(self,
                 latent_size = 64,
                 num_heads = 4,
                 dropout = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = torch.nn.LayerNorm(latent_size)
        self.layer_norm2 = torch.nn.LayerNorm(latent_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
        self.linear = torch.nn.Linear(latent_size,
                                      latent_size)
        self.mha = torch.nn.MultiheadAttention(latent_size,
                                               num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.mlp = MLP(latent_size,
                       dropout=dropout)
    def forward(self, x):
        y = x
        y = self.layer_norm1(y)
        y = self.mha(y,y,y)[0]
        x = y = x + y
        y = self.layer_norm2(y)
        # y = self.linear(y) # Will probably replace this with an MLP block??
        y = self.mlp(y)
        y = self.dropout(y)
        y = self.activation(y)
        return x + y

# Define Trainable Module (Abstract Base Class)
class LightningBoilerplate(pl.LightningModule):
    def __init__(self, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor
        
    def predict_step(self, predict_batch, batch_idx):
        x, y_true = predict_batch
        y_pred = self.predict(x)
        return y_pred, y_true

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('train_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('val_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
        
    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('test_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

# Attach loss, metrics, and optimizer
class MultiClassLightningModule(LightningBoilerplate):
    def __init__(self,
                 num_classes,
                 **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor

        # This creates an accuracy function
        self.network_metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.classification.Accuracy(task='multiclass',
                                                        num_classes=num_classes)
        })
        # This creates a loss function
        self.network_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Attach standardization and augmentation
class StandardizeTransformModule(MultiClassLightningModule):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.standardize = torchvision.transforms.Compose([
            torchvision.transforms.Resize([256]),
            torchvision.transforms.CenterCrop([224]),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225]),
        ])

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(-180.0, 180.0),
                                                translate=(0.1, 0.1),
                                                scale=(0.9, 1.1),
                                                shear=(-10.0, 10.0))#,
            # torchvision.transforms.RandomHorizontalFlip(0.5),
        ])

    def forward(self, x):
        y = x
        y = self.standardize(y)
        if self.training:
            y = self.transform(y)
        return y

class ViTNetwork(StandardizeTransformModule):
    def __init__(self,
                 input_shape,
                 patch_shape,
                 stride_size,
                 output_size,
                 latent_size = 64,
                 num_heads = 4,
                 n_layers = 4,
                 **kwargs):
        super().__init__(num_classes=output_size,**kwargs)
        self.save_hyperparameters()

        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225]),
        ])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(-180.0, 180.0),
                                                translate=(0.1, 0.1),
                                                scale=(0.9, 1.1),
                                                shear=(-10.0, 10.0))#,
            # torchvision.transforms.RandomHorizontalFlip(0.5),
        ])
        
        self.patches = torch.nn.Conv2d(input_shape[1],
                                       latent_size,
                                       patch_shape,
                                       stride_size,
                                       bias=False)
        
        self.position_embedding = SinePositionEmbedding()
        self.transformer_blocks = torch.nn.Sequential(*[
            TransformerBlock(latent_size=latent_size,
                             num_heads=num_heads) for _ in range(n_layers)
        ])
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(latent_size,
                                      output_size)
        
    def forward(self, x):
        y = x
        y = super().forward(y)
        y = self.patches(y)
        y = y.reshape(y.shape[0:2] + (-1,)).permute(0,2,1)
        # y = y + self.position_embedding(torch.arange(0,y.shape[1]).type_as(x).long())
        y = y + self.position_embedding(y)
        y = self.transformer_blocks(y).permute(0,2,1)
        y = self.pooling(y).squeeze()
        y = self.linear(y)
        return y    


# Main i guess #
#########################################################################################


def main():
    batch_size = int(sys.argv[1])
    data_module = EuroSATDataModule(batch_size = batch_size)
    data_module.setup('fit')
    print(data_module.batch_size)

    dl = data_module.train_dataloader()
    batch = next(iter(dl))

    vit_net = ViTNetwork(input_shape=batch[0].shape,
                         patch_shape=(16,16),
                         stride_size=(8,8),
                         output_size=data_module.output_shape[0],
                         latent_size=256,
                         n_layers=6)

    logger = pl.loggers.CSVLogger("logs",
                              name = "VIT-EuroSAT",
                              version = "VIT-EuroSAT")
                              # version = "VIT-EuroSAT-" + os.environ["SLURM_JOB_ID"])

    trainer = pl.Trainer(logger = logger,
                         # max_epochs = 50,
                         max_epochs = 3,
                         enable_progress_bar = True,
                         log_every_n_steps = 0,
                         enable_checkpointing = False,
                         callbacks = [pl.callbacks.TQDMProgressBar(refresh_rate = 50)])

    trainer.fit(vit_net, data_module)

    results = pd.read_csv(logger.log_dir + "/metrics.csv")
    results.to_csv('results-' + sys.argv[1])

    print("Test Accuracy:", results['val_acc'][np.logical_not(np.isnan(results['val_acc']))].iloc[-1])

main()


