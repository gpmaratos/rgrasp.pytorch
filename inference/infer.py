import torch
from torchvision.transforms import Normalize

class Infer:
    """
    Infer. Class which runs the inference module of this project. It
    will take an image, use an existing detector to predict grasps, then
    display them as annotations on the original image. Formats and
    normalizes images in the style specified for pytorch pretrained
    models.
    """

    def __init__(self):
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]
        )
        self.normalize = normalize

    def __call__(self, model, inp):
        xtensor = torch.tensor(inp).permute(2, 0, 1).float()
        xtensor = self.normalize(xtensor)
        xtensor = xtensor.unsqueeze(0)
        pred = model(xtensor)
        import pdb;pdb.set_trace()
