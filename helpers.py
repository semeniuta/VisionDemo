from matplotlib import pyplot as plt


def imshow(im, **kwargs):
    plt.imshow(im, **kwargs)
    plt.axis('off')
    plt.show()


def show_channels(im):

    plt.figure(figsize=(15, 4))

    for ch in range(3):

        plt.subplot(1, 3, ch+1)
        plt.imshow(im[:, :, ch], cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.title('Channel {}'.format(ch+1))
