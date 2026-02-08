# %run setup.py
import sys
sys.argv = [
    "main_notebook_var_load.py",
    "datamodule.model_name=MMGN",
    "datamodule.task=task1",
    "datamodule.sampling_rate=[0.01,0.05]",
    "model=MMGNet"
]
import main_notebook_var_load

# Explicitly call the train function
main_notebook_var_load.train()
global_data_module = main_notebook_var_load.global_data_module

import cartopy.crs as ccrs
import numpy as np
import torch
import torch.nn as nn
from model_interface import MMGNetModule

from tqdm import tqdm

# Add DDP imports
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP

model_path = "./logs/MMGN_task1_s_0.05/checkpoints/last.ckpt"


train_dataloader = global_data_module.train_dataloader
test_dataloader = global_data_module.test_dataloader 

from influence_core import BaseObjective, CGInfluenceModule, BaseInfluenceModule

from torch.utils.data import Dataset, DataLoader
class TimestampDataset(Dataset):
    def __init__(self, dataset, timestamp):
        self.coordinates = dataset[timestamp][0]
        self.temperatures = dataset[timestamp][1]
        self.timestamp = timestamp
        print(self.coordinates.shape)
    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coordinate = self.coordinates[idx]
        temperature = self.temperatures[idx]
        return coordinate[None, ...], temperature[None, ...], self.timestamp

def create_timestamp_dataloader(train_data, timestamp, batch_size=16):
    dataset = TimestampDataset(train_data, timestamp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

loss_fn = torch.nn.L1Loss(reduction="sum")
L2_WEIGHT = 1e-4
class ReconstructionObjective(BaseObjective):
    def train_outputs(self, model, batch):
        return model(batch[0], batch[2])

    def train_loss_on_outputs(self, outputs, batch):
        return loss_fn(outputs, batch[1])

    def train_regularization(self, params):
        return L2_WEIGHT * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        outputs = model(batch[0], batch[2])
        return loss_fn(outputs, batch[1])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using DEVICE: ", DEVICE)

# train_dataloader_0 = create_timestamp_dataloader(global_data_module.train_data, 0)
# test_dataloader_0 = create_timestamp_dataloader(global_data_module.test_data, 0)

import logging
from typing import Callable, Optional

import numpy as np
import scipy.sparse.linalg as L
import torch
from torch import nn
from torch.utils import data
class LiSSAInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the Linear time Stochastic Second-Order Algorithm (LiSSA).

    At a high level, LiSSA estimates an inverse-Hessian vector product
    by using truncated Neumann iterations:

    .. math::
        \mathbf{H}^{-1}\mathbf{v} \approx \frac{1}{R}\sum\limits_{r = 1}^R
        \left(\sigma^{-1}\sum_{t = 1}^{T}(\mathbf{I} - \sigma^{-1}\mathbf{H}_{r, t})^t\mathbf{v}\right)

    Here, :math:`\mathbf{H}` is the risk Hessian matrix and :math:`\mathbf{H}_{r, t}` are
    loss Hessian matrices over batches of training data drawn randomly with replacement (we
    also use a batch size in ``train_loader``). In addition, :math:`\sigma > 0` is a scaling
    factor chosen sufficiently large such that :math:`\sigma^{-1} \mathbf{H} \preceq \mathbf{I}`.

    In practice, we can compute each inner sum recursively. Starting with
    :math:`\mathbf{h}_{r, 0} = \mathbf{v}`, we can iteratively update for :math:`T` steps:

    .. math::
        \mathbf{h}_{r, t} = \mathbf{v} + \mathbf{h}_{r, t - 1} - \sigma^{-1}\mathbf{H}_{r, t}\mathbf{h}_{r, t - 1}

    where :math:`\mathbf{h}_{r, T}` will be equal to the :math:`r`-th inner sum.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        repeat: the number of trials :math:`R`.
        depth: the recurrence depth :math:`T`.
        scale: the scaling factor :math:`\sigma`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        debug_callback: a callback function which is passed in :math:`(r, t, \mathbf{h}_{r, t})`
            at each recurrence step.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: DataLoader,
            test_loader: DataLoader,
            device: torch.device,
            damp: float,
            repeat: int,
            depth: int,
            scale: float,
            gnh: bool = False,
            debug_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.repeat = repeat
        self.depth = depth
        self.scale = scale
        self.debug_callback = debug_callback

    def inverse_hvp(self, vec):

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        ihvp = 0.0

        for r in range(self.repeat):

            h_est = vec.clone()

            for t, (batch, _) in enumerate(self._loader_wrapper(sample_n_batches=self.depth, train=True)):

                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=h_est, gnh=self.gnh)

                with torch.no_grad():
                    hvp_batch = hvp_batch + self.damp * h_est
                    h_est = vec + h_est - hvp_batch / self.scale

                if self.debug_callback is not None:
                    self.debug_callback(r, t, h_est)

            ihvp = ihvp + h_est / self.scale

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return ihvp / self.repeat

# DDP setup functions
def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()

def compute_influences_ddp(rank, world_size, test_idxs, model_path, train_dataloader_0, test_dataloader_0, all_train_idxs):
    """Compute influences using DDP"""
    setup_ddp(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Split test indices for this rank
    chunk_size = len(test_idxs) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(test_idxs)
    test_idxs_chunk = test_idxs[start_idx:end_idx]
    
    print(f"Rank {rank}: Processing test indices {start_idx} to {end_idx-1} ({len(test_idxs_chunk)} samples)")
    
    # Load model
    MMGNetModel = MMGNetModule.load_from_checkpoint(model_path, map_location=device)
    model = MMGNetModel.model.to(device)
    
    # Create module
    module = LiSSAInfluenceModule(
        model=model,
        objective=ReconstructionObjective(),
        train_loader=train_dataloader_0,
        test_loader=test_dataloader_0,
        device=device,
        damp=0.001,
        repeat=5,
        depth=10,
        scale=10
    )
    
    # Compute influences for this chunk
    influences_chunk = []
    for test_idx in tqdm(test_idxs_chunk, desc=f"Rank {rank} Computing Influences"):
        influences = module.influences(train_idxs=all_train_idxs, test_idxs=[test_idx])
        influences_chunk.append(influences)
    
    # Convert all tensors to numpy arrays
    influences_chunk_np = [x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x) for x in influences_chunk]
    # Save to .npy file
    filename = f"influences_{start_idx}_to_{end_idx-1}.npy"
    np.save(filename, influences_chunk_np, allow_pickle=True)
    print(f"Rank {rank}: Influence results saved to {filename}")
    
    cleanup_ddp()

def main():
    """Main function to run DDP influence computation"""
    train_dataloader_0 = create_timestamp_dataloader(global_data_module.train_data, 0)
    test_dataloader_0 = create_timestamp_dataloader(global_data_module.test_data, 0)
    
    all_train_idxs = list(range(len(train_dataloader_0.dataset)))
    test_idxs = list(range(len(test_dataloader_0.dataset)))
    # test_idxs = list(range(96))
    
    num_gpus = 8
    print(f"Starting DDP influence computation on {num_gpus} GPUs")
    print(f"Total test samples: {len(test_idxs)}")
    print(f"Samples per GPU: {len(test_idxs) // num_gpus}")
    
    # Run DDP
    mp.spawn(
        compute_influences_ddp,
        args=(num_gpus, test_idxs, model_path, train_dataloader_0, test_dataloader_0, all_train_idxs),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()

