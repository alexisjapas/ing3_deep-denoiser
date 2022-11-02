from random import uniform, randint
import numpy as np


def salt_and_pepper(image, density):
    # Transforms and compute image data
    image = image.astype(float)
    max_image = np.max(image)
    min_image = np.min(image)

    # Browse all the pixels of the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # For each, replaces its value with either its min or max
            if density * 100 > float(uniform(0, 100)):
                if randint(0, 1):
                    image[y, x] = max_image
                else:
                    image[y, x] = min_image

    return image


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib import image as mpimg


    image = mpimg.imread("/home/obergam/Data/flir/images_thermal_train/video-zNFzcc9wW8XB4QwTa-frame-011398-BnmWrvMFmpcrPWRdk.jpg")
    plt.figure(1)
    plt.imshow(image)

    noisy_image = salt_and_pepper(image, 0.1)
    plt.figure(2)
    plt.imshow(noisy_image)
    plt.show()
