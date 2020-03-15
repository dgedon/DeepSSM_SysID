import torch

from models import DynamicModel
import torch.optim as optim
import os.path


class ModelState:
    """
    Container for all model related parameters and optimizer

    model
    optimizer
    """

    def __init__(self, seed, nu, ny, model, options, **kwargs):
        torch.manual_seed(seed)

        self.model = DynamicModel(model, nu, ny, options, **kwargs)

        # Optimization parameters
        self.optimizer = getattr(optim, options['optim'])(self.model.parameters(), lr=options['train_options'].init_lr)

    def load_model(self, path, name='model.pt'):
        file = path if os.path.isfile(path) else os.path.join(path, name)
        try:
            ckpt = torch.load(file, map_location=lambda storage, loc: storage)
        except NotADirectoryError:
            raise Exception("Could not find model: " + file)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt['epoch']
        return epoch

    def save_model(self, epoch, vloss, elapsed_time,  path, name='model.pt'):
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'vloss': vloss,
                'elapsed_time': elapsed_time,
            },
            os.path.join(path, name))
