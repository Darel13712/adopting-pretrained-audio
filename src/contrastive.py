import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ContrastModel(nn.Module):
    def __init__(self, pretrained_content, pretrained_collaborative, hidden_dim=768):
        super(ContrastModel, self).__init__()

        self.hidden_dim = hidden_dim

        pretrained_content = torch.from_numpy(pretrained_content)
        self.pretrained_content = nn.Embedding.from_pretrained(pretrained_content, freeze=True)
        content_dim = self.pretrained_content.weight.shape[1]

        pretrained_collaborative = torch.from_numpy(pretrained_collaborative)
        self.pretrained_collaborative = nn.Embedding.from_pretrained(pretrained_collaborative, freeze=True)
        collaborative_dim = self.pretrained_collaborative.weight.shape[1]
        
        self.content_model = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.collaborative_model = nn.Sequential(
            nn.Linear(collaborative_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, indices):
        content_embeds = self.pretrained_content(indices)
        content_embeds = self.content_model(content_embeds)
        content_embeds = F.normalize(content_embeds, p=2, dim=-1)

        collaborative_embeds = self.pretrained_collaborative(indices)
        collaborative_embeds = self.collaborative_model(collaborative_embeds)
        collaborative_embeds = F.normalize(collaborative_embeds, p=2, dim=-1)


        return content_embeds, collaborative_embeds

    def extract_embeddings(self):
        """
        Extract normalized embeddings
        """
        # Extract raw embeddings
        content = self.pretrained_content.weight.data
        collaborative = self.pretrained_collaborative.weight.data

        # Process through the model's sequential layers
        with torch.no_grad():
            content = self.content_model(content).cpu().numpy()
            collaborative = self.collaborative_model(collaborative).cpu().numpy()

        return content, collaborative



SMALL_NUM = np.log(1e-45)


class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)


class SDCL(object):
    """
    Symmetrical Decoupled Contrastive Loss. because why not?
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(SDCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance, torch.mm(z2, z2.t())), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 3)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

class SDCLW(DCL):
    """
    Symmetrical Decoupled Contrastive Loss with negative von Mises-Fisher weighting
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(SDCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)

class ADCL(object):
    """
    Alternative Decoupled Contrastive Loss similar to https://arxiv.org/pdf/2501.01108
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(ADCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = cross_view_distance / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class ADCLW(ADCL):
    """
    Alternative Decoupled Contrastive Loss with negative von Mises-Fisher weighting similar to https://arxiv.org/pdf/2501.01108
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(ADCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)

