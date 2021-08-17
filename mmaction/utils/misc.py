import ctypes
import os
import random
import string


def get_random_string(length=15):
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Default: 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id():
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir():
    """Get shm dir for temporary usage."""
    return '/dev/shm'


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def add_suffix(inputs, suffix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        suffix (str): The suffix to add.

    Returns:

        dict: The dict with keys updated with ``suffix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{name}.{suffix}'] = value

    return outputs


def terminal_is_available():
    for key in os.environ:
        if key.startswith('KUBERNETES'):
            return False
    return True


def tuple_divide(input_tuple, divisor):
    return tuple(i // divisor for i in input_tuple)
