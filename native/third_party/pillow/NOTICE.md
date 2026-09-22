The RGB tracking-frame bicubic resampler in `native/src/tracking_preprocess.cpp`
preserves the coefficient, intermediate-byte and fixed-point rounding behavior
of Pillow 12.2.0's `src/libImaging/Resample.c`. The original tracker loads JPEG
sequences through Pillow; generic tensor bicubic interpolation differs at edges.

Source: https://github.com/python-pillow/Pillow/blob/12.2.0/src/libImaging/Resample.c
Pillow's HPND license/copyright/permission notice is included as LICENSE.txt.
The native implementation uses no Pillow or Python runtime.
