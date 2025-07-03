import jax
import jax.numpy as jnp


def truncated_fft_2x(u: jax.Array) -> jax.Array:
    uhat = jnp.fft.fftshift(jnp.fft.fft(u))
    (k,) = uhat.shape
    final_size = (k + 1) // 2
    return jnp.fft.ifftshift(uhat[final_size // 2 : (-final_size + 1) // 2]) / 2


def padded_ifft_2x(uhat: jax.Array) -> jax.Array:
    (n,) = uhat.shape
    final_size = n + 2 * (n // 2)
    added = n // 2
    smoothed = jnp.pad(jnp.fft.fftshift(uhat), (added, added))
    assert smoothed.shape == (final_size,), "incorrect padded shape"
    return 2 * jnp.fft.ifft(jnp.fft.ifftshift(smoothed))


def truncated_fft_2x_2D(u: jax.Array) -> jax.Array:
    uhat = jnp.fft.fftshift(jnp.fft.fftn(u))
    kx, ky = uhat.shape
    final_size_x, final_size_y = (kx + 1) // 2, (ky + 1) // 2
    return (
        jnp.fft.ifftshift(
            uhat[
                final_size_x // 2 : (-final_size_x + 1) // 2,
                final_size_y // 2 : (-final_size_y + 1) // 2,
            ]
        )
        / 4
    )


def padded_ifft_2x_2D(uhat: jax.Array) -> jax.Array:
    nx, ny = uhat.shape
    final_size_x, final_size_y = nx + 2 * (nx // 2), ny + 2 * (ny // 2)
    added_x, added_y = nx // 2, ny // 2
    smoothed = jnp.pad(jnp.fft.fftshift(uhat), ((added_x, added_x), (added_y, added_y)))
    assert smoothed.shape == (final_size_x, final_size_y), "incorrect padded shape"
    return jnp.fft.ifftn(jnp.fft.ifftshift(smoothed)) * 4