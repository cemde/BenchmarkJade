from typing import List, Tuple
import platform
import os
from typing import Any, Dict
import yaml
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
import lightning as L

def print_system_info(fabric: L.Fabric) -> None:
    if fabric.local_rank != 0:
        return
    
    try:
        import psutil
    except ImportError:
        print("psutil not installed. Cannot print system information.")
        return
    try:
        # Number of physical and virtual cores
        physical_cores = psutil.cpu_count(logical=False)
        virtual_cores = psutil.cpu_count(logical=True)

        # CPU model name
        cpu_model_name = platform.processor()

        # Total memory in GB
        total_memory = psutil.virtual_memory().total / (1024 ** 3)

        # cuda driver
        try:
            cuda_driver = os.popen('nvidia-smi --query-gpu=driver_version --format=csv,noheader').read().split('\n')[0]
        except:
            cuda_driver = "N/A"

        print("-" * 50)
        print("System Information")
        print("-" * 50)
        print(f"Physical CPU Cores: {physical_cores}")
        print(f"Virtual CPU Cores:  {virtual_cores}")
        print(f"CPU Model:          {cpu_model_name}")
        print(f"Total Memory:       {total_memory:.2f} GB")
        print("-" * 30)
        print(f"torch version:      {torch.__version__}")
        print(f"cuda version:       {torch.version.cuda}")
        print(f"driver version:     {cuda_driver}")
        print(f"Devices:            {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")

    except Exception as e:
        print(f"Error while printing system information: {e}")


def print_args(fabric: L.Fabric, args: argparse.Namespace) -> None:
    if fabric.local_rank != 0:
        return
    fabric.print("-"*50)
    fabric.print("Arguments")
    fabric.print("-"*50)
    for arg in vars(args):
        fabric.print(f"{arg}: {getattr(args, arg)}")
    fabric.print("")


def print_benchmark_results(fabric: L.Fabric, args: argparse.Namespace, data_loading_times: List[float], data_preprocessing_times: List[float], forward_pass_times: List[float], backward_pass_times: List[float], loop_time: float) -> None:
    fabric.print("-"*50)
    fabric.print("Benchmark Results")
    fabric.print("-"*50)
    fabric.print(f"Devices used:                    {fabric.world_size} x {torch.cuda.get_device_name(fabric.device)}")
    fabric.print(f"Average Data Loading Time:       {sum(data_loading_times)/len(data_loading_times):.4f} seconds")
    fabric.print(f"Average Data Preprocessing Time: {sum(data_preprocessing_times)/len(data_preprocessing_times):.4f} seconds")
    fabric.print(f"Average Forward Pass Time:       {sum(forward_pass_times)/len(forward_pass_times):.4f} seconds")
    fabric.print(f"Average Backward Pass Time:      {sum(backward_pass_times)/len(backward_pass_times):.4f} seconds")
    fabric.print(f"Total Script Time:               {loop_time:.4f} seconds")


class ClusterManager:
    def __init__(self, name: str = None, auto: bool = True):
        """Creates a ClusterManager object that can automatically configure multiple clusters.

        Args:
            name (str): Name of the cluster as given in YAML.
            auto (bool, optional): Whether the cluster should be identified automatically from the linux environment variables. Defaults to True.

        Raises:
            OSError: YAML config file not found.
            NotImplementedError: Cluster ID not found in YAML config file.
        """
        if auto:
            sys_name = os.getenv("CLUSTER_NAME")
            if sys_name is None:
                raise OSError("CLUSTER_NAME not found in environment variables. Autoselecting system failed.")
            else:
                self.name = sys_name
        else:
            self.name = name

        config_file = "config/system.yaml"
        with open(config_file) as file:
            self._configs = yaml.load(file, Loader=yaml.FullLoader)

        if self.name not in self._configs.keys():
            raise NotImplementedError(f"System {self.name} not implemented in '{config_file}'")

        self._configs = self._configs[self.name]

    @property
    def project_dir(self) -> str:
        return self._configs["PROJECT_DIR"]

    @property
    def num_workers(self) -> int:
        return self._configs["NUM_WORKERS"]

    @property
    def data_dir(self) -> str:
        return self._configs["DATA_DIR"]

    @property
    def log_dir(self) -> str:
        return self._configs["LOG_DIR"]

    @property
    def artifact_dir(self) -> str:
        return self._configs["ARTIFACTS_DIR"]

    @property
    def network(self):
        return self._configs["NETWORK"]

    @property
    def use_GPU(self) -> bool:
        return self._configs["USE_GPU"]

    @property
    def get_pid(self) -> int:
        try:
            return os.environ["SLURM_JOB_ID"]
        except KeyError:
            return os.getpid()

    def to_dict(self) -> Dict[str, Any]:
        return {k.lower(): v for k, v in self._configs.items()}

def setup(rank, world_size):
    dist.init_process_group(
        "nccl",         # backend, "nccl" is recommended for GPU training
        rank=rank,      # the current rank of the process
        world_size=world_size  # the total number of processes
    )
    torch.cuda.set_device(rank)  # set the current GPU device

def cleanup():
    dist.destroy_process_group()

DATASETS = ["imagenet", "dummy"]

def prepare(dataset: str, architecture: str, batch_size: int, num_workers: int, devices: List[int], cluster: ClusterManager, strategy: str = "dp", precision: str = "fp32"):

    if architecture == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError(f"Architecture {architecture} not implemented.")


    # Define the ImageNet transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset == "imagenet":
        train_dataset = datasets.ImageNet(root=os.path.join(cluster.data_dir, "ImageNet"), split='val', transform=transform)
        val_dataset = datasets.ImageNet(root=os.path.join(cluster.data_dir, "ImageNet"), split='val', transform=transform)

    elif dataset == "dummy":
        n_data = 50000
        train_dataset = datasets.FakeData(size=n_data, image_size=(3, 224, 224), num_classes=1000, transform=transform)
        val_dataset = datasets.FakeData(size=n_data, image_size=(3, 224, 224), num_classes=1000, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dataset == "imagenet"))
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dataset == "imagenet"))

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    # launch the distributed training
    fabric = L.Fabric(accelerator="cuda", strategy=strategy, devices=devices, precision=precision)
    fabric.launch()

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    return fabric, model, optimizer, train_dataset, val_dataset, train_loader, val_loader
