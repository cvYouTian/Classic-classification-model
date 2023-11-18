from types import NoneType
from typing import Union


def auto_padding(kernel_size: Union[int, tuple, list],
                 padding: NoneType):
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]

    return padding


if __file__ == "__main__":
    ...