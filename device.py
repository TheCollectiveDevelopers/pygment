import torch

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" #type: ignore
