from math import sqrt

from dall_e.decoder import DecoderBlock
from dall_e.encoder import EncoderBlock
from taming.modules.diffusionmodules.model import ResnetBlock
import torch

from dalle_pytorch import distributed_utils
from dalle_pytorch import DALLE, DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024
from dalle_pytorch.dalle_pytorch import ResBlock


class ResBlockGradScaler:
    def __init__(self):
        if distributed_utils.is_distributed:
            self._M = distributed_utils.backend.get_world_size()
        else:
            self._M = 1
        self._min_grad_scale_window = 125
        self._min_grad_scale = self._M * 2**7
        self._max_grad_scale = self._M * 2**24

        self._grad_scale_window = self._min_grad_scale_window
        self.grad_scale = torch.tensor(self._M * 2**13, dtype=torch.float32)

    def _clamp_grad_scale(self):
        self.grad_scale = torch.clamp(
            self.grad_scale, self._min_grad_scale, self._max_grad_scale)

    @staticmethod
    def _zero_nonfinite(tensor):
        nonfinite = ~torch.isfinite(tensor)
        tensor[nonfinite] = 0
        return tensor

    def scale_grad(self, grad_input, grad_output):
        new_grad_input = grad_input
        if not isinstance(grad_input, tuple):
            new_grad_input = tuple(new_grad_input)

        new_grad_input = tuple(map(lambda x: x * self.grad_scale,
                                   new_grad_input))
        all_finite = all(map(lambda x: torch.isfinite(x).all(),
                             new_grad_input))

        if all_finite:
            self.grad_scale *= 2**(1 / 1000)
            self._clamp_grad_scale()
            self._grad_scale_window += 1
        elif self._grad_scale_window >= self._min_grad_scale_window:
            new_grad_input = tuple(map(lambda x: self._zero_nonfinite(x),
                                       new_grad_input))

            self.grad_scale /= sqrt(2)
            self._clamp_grad_scale()
            self._grad_scale_window = 0

        if not isinstance(grad_input, tuple):
            return new_grad_input[0]
        return new_grad_input

    def __call__(self, _module, grad_input, grad_output):
        self.scale_grad(grad_input, grad_output)


def add_resblock_grad_scalers(module):
    torch_ver = list(map(int, torch.__version__.split('.')[:2]))
    torch_ver_over_1_7 = torch_ver[0] >= 1 and torch_ver[1] > 7
    if torch_ver_over_1_7:
        register_backward_hook_func = \
            torch.nn.Module.register_full_backward_hook
    else:
        register_backward_hook_func = torch.nn.Module.register_backward_hook

    if isinstance(module, DALLE):
        # We are only interested in the VAE (which contains
        # the resblocks)
        module = module.vae

    if isinstance(module, DiscreteVAE):
        resblock_type = ResBlock
    elif isinstance(module, OpenAIDiscreteVAE):
        resblock_type = (DecoderBlock, EncoderBlock)
    elif isinstance(module, VQGanVAE1024):
        resblock_type = ResnetBlock
    else:
        raise ValueError('unknown VAE')

    for submodule in module.modules():
        if not isinstance(submodule, ResBlock):
            continue
        register_backward_hook_func(submodule, ResBlockGradScaler())
