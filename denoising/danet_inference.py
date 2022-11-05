import numpy as np
import torch
from skimage import img_as_float32
import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '../third_patry/DANet/')
)

from denoising.window_inference import inference_tiling_intersected
from networks import UNetD
from networks.transformer import Uformer


class DANetInference(object):
    net = None

    def __init__(self, net_arch: str, net_weights: str, device: str = 'cpu'):
        # load the pretrained model
        if net_arch == 't':
            self.net = Uformer(img_size=128)
            self.net.load_state_dict(
                torch.load(net_weights, map_location='cpu')['D'])
        else:
            self.net = UNetD(3, wf=32, depth=5)
            if net_arch == 'danet':
                self.net.load_state_dict(torch.load(net_weights, map_location='cpu')['D'])
            else:
                self.net.load_state_dict(torch.load(net_weights, map_location='cpu'))

        self.net = self.net.to(device)
        self.net.eval()

        self.device = device

    def inference(self, inp_tensor: torch.Tensor) -> torch.Tensor:
        outputs = inp_tensor - self.net(inp_tensor)
        outputs.clamp_(0.0, 1.0)
        return outputs

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Call method to denoise
        Args:
            image: RGB image array with uint8 dtype

        Returns:
            Denoise RGB image
        """
        inputs = torch.from_numpy(
            img_as_float32(image).transpose([2, 0, 1])
        ).to(self.device)
        with torch.no_grad():
            out = inference_tiling_intersected(inputs, self.inference, tile_size=128)

        im_denoise = (out.cpu().numpy().transpose(
            [1, 2, 0]) * 255).astype(np.uint8)

        return im_denoise


if __name__ == '__main__':
    import cv2

    image = cv2.cvtColor(
        cv2.imread(
            '../data/images_dataset/sony_calibrated/DSC01215.JPG',
            cv2.IMREAD_COLOR
        ),
        cv2.COLOR_BGR2RGB
    )
    w = '../third_patry/DANet/model_states/DANet.pt'
    model = DANetInference(net_arch='danet', net_weights=w, device='cuda')
    denoise_image = model(image)
    cv2.imwrite('../data/denoised_15.jpg', denoise_image)
