import jax
import jax.numpy as jnp


def test_import():
    import soundersim

    assert soundersim.__version__


def test_jax_backend_cpu():
    assert jax.default_backend() == "cpu"


def test_jax_jit():
    f = jax.jit(lambda x: x * 2 + 1)
    assert float(f(jnp.array(3.0))) == 7.0
