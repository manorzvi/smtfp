from typing import Tuple, Union
import numpy as np
import pandas as pd
import struct
import torch
from loguru import logger

            
class PackedFP32:
    def __init__(self) -> None:
        pass

    def from_fp32(self, x: Union[float, np.float32, np.float32]) -> Tuple[str, str, str]:
        assert isinstance(x, float) or isinstance(x, np.float32) or isinstance(x, np.float32) or isinstance(x, torch.FloatTensor)
        x = struct.pack('!f', x)  # use '!f' for big-endian byte order
        x = [bin(c)[2:].rjust(8, '0') for c in x]
        x = ''.join(x)
        sign = x[0]
        exponent = x[1:9]
        mantissa = x[9:]
        return sign, exponent, mantissa

    def to_fp32(self, sign: str, exponent: str, mantissa: str) -> np.float32:
        """
        https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        """
        assert isinstance(sign, str) and len(sign) == 1 and \
            isinstance(exponent, str) and len(exponent) == 8 and \
            isinstance(mantissa, str) and len(mantissa) == 23
        dec_sign = (-1) ** int(sign, 2)
        dec_exponent = int(exponent, 2)
        dec_mantissa = 1.0
        for i, b in enumerate(mantissa, start=1):
            dec_mantissa += int(b, 2) * 1/(2 ** i)
        npfp32 = np.float32(dec_sign * 2 ** (dec_exponent - 127) * dec_mantissa)
        return npfp32
    
    def floor_mantissa(self, x: Union[float, float32, np.float32]) -> np.float32:
        sign, exponent, _ = self.from_fp32(x)
        mantissa = '00000000000000000000000'
        x = self.to_fp32(sign, exponent, mantissa)
        return x
    
    def ceil_mantissa(self, x: Union[float, float32, np.float32]) -> np.float32:
        sign, exponent, _ = self.from_fp32(x)
        mantissa = '00000000000000000000000'
        exponent = int(exponent, 2)
        exponent += 1
        exponent = bin(exponent)[2:].rjust(8, '0')
        x = self.to_fp32(sign, exponent, mantissa)
        return x
    
    def zero_mantissa_to_closest_exponent(self, x: Union[float, float32, np.float32]) -> np.float32:
        _, _, mantissa = self.from_fp32(x)
        if mantissa[0] == '0':
            x = self.floor_mantissa(x)
        else:
            x = self.ceil_mantissa(x)
        return x


def test():
    """
    Use https://www.h-schmidt.net/FloatConverter/IEEE754.html for manual testing
    """
    X = np.random.normal(scale=100, size=10000).astype(np.float32)
    X_df = pd.DataFrame(X)
    print(X_df.describe())
    err_list = []
    for x in X:
        print('{:<20}'.format(x), end=' | ')
        sign, exponent, mantissa = PackedFP32().from_fp32(x)
        print(f'{sign} {exponent} {mantissa}', end=' | ')
        x2 = PackedFP32().to_fp32(sign, exponent, mantissa)
        print('{:<20}'.format(x2), end=' || ')
        # x3 = PackedFP32().floor_mantissa(x)
        # x3 = PackedFP32().ceil_mantissa(x)
        x3 = PackedFP32().zero_mantissa_to_closest_exponent(x)
        print('{:<10}'.format(x3), end=' | ')
        signx3, exponentx3, mantissax3 = PackedFP32().from_fp32(x3)
        print(f'{signx3} {exponentx3} {mantissax3}', end=' || ')
        err = np.abs(x3-x)
        print('{:<10}'.format(err))
        err_list.append(err)
    print('-'*200)
    print(np.asarray(err_list).mean())

        


if __name__ == '__main__':
    np.random.seed(42)
    test()