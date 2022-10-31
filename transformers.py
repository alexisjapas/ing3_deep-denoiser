from random import randint


def random_crop(image, y_dim, x_dim):
    y_top = randint(0, image.shape[0] - y_dim)
    y_bot = y_top + y_dim
    x_left = randint(0, image.shape[1] - x_dim)
    x_right = x_left + x_dim
    return image[y_top:y_bot, x_left:x_right]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib import image as mpimg


    image = mpimg.imread("/home/obergam/Data/flir/images_thermal_train/video-zNFzcc9wW8XB4QwTa-frame-011398-BnmWrvMFmpcrPWRdk.jpg")
    plt.figure(0)
    plt.imshow(image)
    cropped_image = random_crop(image, 82, 82)
    print(image.shape, cropped_image.shape)
    plt.figure(1)
    plt.imshow(cropped_image)
    plt.show()
