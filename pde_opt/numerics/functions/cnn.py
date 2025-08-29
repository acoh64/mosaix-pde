
from typing import Callable, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn

Array = jax.Array

class PeriodicConvBlock(eqx.Module):
    """Conv2d -> activation with periodic padding. Translation-equivariant."""
    conv: nn.Conv2d
    act: Callable[[Array], Array]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        act: Callable[[Array], Array] = jax.nn.gelu,
        *,
        key: Array,
    ):
        # SAME keeps spatial dims; CIRCULAR enforces periodic BCs.
        # Input/Output shapes: (C, H, W) -> (C', H, W)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="SAME",          # keeps spatial dims equal to input
            padding_mode="CIRCULAR", # periodic wrap-around padding
            use_bias=True,
            key=key,
        )
        self.act = act

    def __call__(self, x: Array) -> Array:
        return self.act(self.conv(x))


class PeriodicCNN(eqx.Module):
    """Stack of periodic conv blocks; final conv returns requested channels.
    
    Translation-equivariant on a torus so long as stride=1 and only pointwise
    nonlinearities are used. Accepts (C,H,W) or (B,C,H,W); returns same spatial size.
    """
    layers: Tuple[eqx.Module, ...]  # blocks + final conv (no activation)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int] = (32, 64, 64),
        out_channels: Union[int, None] = None,
        kernel_size: int = 3,
        act: Callable[[Array], Array] = jax.nn.gelu,
        *,
        key: Array,
    ):
        assert kernel_size % 2 == 1, "Use odd kernels to avoid off-by-one alignment."
        keys = jax.random.split(key, len(hidden_channels) + 1)

        if out_channels is None:
            out_channels = in_channels

        blocks = []
        c_prev = in_channels
        for i, c_next in enumerate(hidden_channels):
            blocks.append(
                PeriodicConvBlock(
                    c_prev, c_next, kernel_size=kernel_size, act=act, key=keys[i]
                )
            )
            c_prev = c_next

        # Final linear conv back to out_channels (no activation)
        final_conv = nn.Conv2d(
            in_channels=c_prev,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="SAME",
            padding_mode="CIRCULAR",
            use_bias=True,
            key=keys[-1],
        )
        self.layers = tuple([*blocks, final_conv])

    def _forward_single(self, x: Array) -> Array:
        # x: (C, H, W)
        for layer in self.layers[:-1]:
            x = layer(x)  # Conv + activation
        x = self.layers[-1](x)  # final Conv, no activation
        return x

    def __call__(self, x: Array) -> Array:
        return self._forward_single(x[None])[0]
        