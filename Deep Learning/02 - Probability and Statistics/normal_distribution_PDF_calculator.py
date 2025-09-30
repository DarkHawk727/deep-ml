import math


def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    # Your code here
    val = (
        1.0
        / math.sqrt(2.0 * math.pi * math.pow(std_dev, 2))
        * math.exp(-math.pow(x - mean, 2) / (2.0 * math.pow(std_dev, 2)))
    )
    return round(val, 5)
