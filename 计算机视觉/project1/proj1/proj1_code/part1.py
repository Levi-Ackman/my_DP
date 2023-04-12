#!/usr/bin/python3

import numpy as np


def create_Gaussian_kernel(cutoff_frequency):
    """
        Returns a 2D Gaussian kernel using the specified filter size standard
        deviation and cutoff frequency.

        The kernel should have:
        - shape (k, k) where k = cutoff_frequency * 4 + 1
        - mean = floor(k / 2)
        - standard deviation = cutoff_frequency
        - values that sum to 1

        Args:
        - cutoff_frequency: an int controlling how much low frequency to leave in
          the image.
        Returns:
        - kernel: numpy nd-array of shape (k, k)

        HINT:
        - The 2D Gaussian kernel here can be calculated as the outer product of two
          vectors with values populated from evaluating the 1D Gaussian PDF at each
          corrdinate.
    """
    k = cutoff_frequency * 4 + 1
    mean = np.floor(k / 2)
    std = cutoff_frequency
    gauss_1d = np.zeros((k, 1))

    total = 0
    index = 0
    for x in range(-int(mean), int(mean) + 1):
        x1 = 1 / np.sqrt(2 * np.pi * std ** 2)
        x2 = np.exp(-(x ** 2) / (2 * std ** 2))
        g = x1 * x2
        gauss_1d[index] = g
        index += 1
        total += g
    kernel = np.outer(gauss_1d, gauss_1d) / total ** 2
    return kernel


def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of shape (m, n, c)
    - filter: numpy nd-array of shape (k, j)
    Returns
    - filtered_image: numpy nd-array of shape (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using OpenCV or similar to do the
    filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that the TAs can verify
    your code works.
    """
    # finish the function
    filtered_image = np.asarray([0])

    ##################
    # Your code here #
    pad_x = (filter.shape[0] - 1) // 2
    pad_y = (filter.shape[1] - 1) // 2
    # 由于要输出原尺寸图像，因此先对图像进行padding操作，再卷积
    image_pad = np.pad(image, (
        (pad_x, pad_x),
        (pad_y, pad_y),
        (0, 0)), 'constant')
    # 计算滤波后图像的尺寸
    filtered_image_height = image_pad.shape[0] - filter.shape[0] + 1
    filtered_image_width = image_pad.shape[1] - filter.shape[1] + 1
    filtered_image = np.zeros([filtered_image_height, filtered_image_width, image.shape[2]])
    # 进行卷积操作
    for k in range(image.shape[2]):
        for i in range(filtered_image_height):
            for j in range(filtered_image_width):
                filtered_image[i, j, k] = np.sum(
                    np.multiply(image_pad[i:i + filter.shape[0], j:j + filter.shape[1], k], filter))
    # raise NotImplementedError('my_imfilter function in helpers.py needs to be implemented')
    ##################

    return filtered_image





    pass


def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (x, y)
    Returns
    - low_frequencies: numpy nd-array of shape (m, n, c)
    - high_frequencies: numpy nd-array of shape (m, n, c)
    - hybrid_image: numpy nd-array of shape (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are between
      0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    image1_low = my_imfilter(image1, filter)
    image2_low = my_imfilter(image2, filter)
    image2_high = image2 - image2_low

    hybrid_image = np.clip(image1_low + image2_high, 0, 1)

    low_frequencies = image1_low
    high_frequencies = image2_high

    return low_frequencies, high_frequencies, hybrid_image
