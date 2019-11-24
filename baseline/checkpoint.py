"""Checkpoint"""
import torch

class Chechpoint(object):
    """Checkpoint class"""
    @staticmethod
    def save(model):
        """Save model using name"""
        name = f'{model.name}.pt'
        torch.save(model.state_dict(), name)

    @staticmethod
    def load(model):
        """Load model using name"""
        name = f'{model.name}.pt'
        model.load_state_dict(torch.load(name))
        return model
