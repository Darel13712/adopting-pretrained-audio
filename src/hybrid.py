import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ELSAHybridModel(nn.Module):
    def __init__(self, pretrained, hidden_dim=768, device=None):
        super(ELSAHybridModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.__nmse = NMSELoss()
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

        self.pretrained = torch.from_numpy(pretrained).to(self.device)
        # self.A = nn.Embedding.from_pretrained(pretrained, freeze=True)
        content_dim = self.pretrained.shape[1]

        self.model = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, x):
        A = F.normalize(self.model(self.pretrained), p=2, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return xAAT - x


    def extract_embeddings(self):
        """
        Extract normalized embeddings
        """

        # Process through the model's sequential layers
        with torch.no_grad():
            content = F.normalize(self.model(self.pretrained), p=2, dim=-1).cpu().numpy()

        return content


class NMSELoss(torch.nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(input, dim=-1),
            torch.nn.functional.normalize(target, dim=-1),
            reduction='mean'
        )