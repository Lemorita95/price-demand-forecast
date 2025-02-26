import matplotlib.pyplot as plt
PX = 1/plt.rcParams['figure.dpi']  # pixel in inches

from auxiliary.helpers import os, np, IMAGES_FOLDER


def plot_1_axis(x1, y1, x2, y2, **kwargs):
    fig, ax = plt.subplots(figsize=(1280*PX, 720*PX))

    color1 = kwargs.pop('color1', '#000000')
    color2 = kwargs.pop('color2', '#000000')
    title = kwargs.pop('title', 'Title')
    x_label = kwargs.pop('x_label', 'X-Label')
    y_label = kwargs.pop('y_label', 'Y1-Label')
    y1_legend = kwargs.pop('y1_legend', 'Y1-Legend')
    y2_legend = kwargs.pop('y2_legend', 'Y2-Legend')
    fontsize = kwargs.pop('fontsize', 12)
    x_lim = kwargs.pop('x_lim', (np.min(np.concatenate((x1, x2))), np.max(np.concatenate((x1, x2)))))
    y_lim = kwargs.pop('y_lim', (np.min(np.concatenate((y1.reshape(-1, 1), y2.reshape(-1, 1)))), np.max(np.concatenate((y1.reshape(-1, 1), y2.reshape(-1, 1))))))
    filename = kwargs.pop('filename', 'not-save')

    color = color1
    ax.set_ylabel(y_label, color='#000000', fontsize = fontsize)
    ax.plot(x1, y1, color=color, label=y1_legend, linewidth=0.50)
    # ax.tick_params(axis='y', labelcolor=color)

    color = color2
    ax.plot(x2, y2, color=color, label=y2_legend, linewidth=1.00)

    ax.set_xlabel(x_label)
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

    ax = plt.gca()
    ax.set_xlim(x_lim)
    y_delta = y_lim[1] - y_lim[0]
    y_lim = (y_lim[0] - y_delta/20, y_lim[1] + y_delta/20)
    ax.set_ylim(y_lim)
    ax.set_title(title)

    plt.grid(alpha=0.25)
    fig.tight_layout()

    # save fig if have filename
    if filename != 'not-save':
        plt.savefig(os.path.join(IMAGES_FOLDER, f'{filename}.png'))
    plt.show()


def plot_2_axis(x1, y1, x2, y2, **kwargs):
    color1 = kwargs.pop('color1', '#000000')
    color2 = kwargs.pop('color2', '#000000')
    title = kwargs.pop('title', 'Title')
    x_label = kwargs.pop('x_label', 'X-Label')
    y1_label = kwargs.pop('y1_label', 'Y1-Label')
    y2_label = kwargs.pop('y2_label', 'Y2-Label')
    y1_legend = kwargs.pop('y1_legend', y1_label)
    y2_legend = kwargs.pop('y2_legend', y2_label)
    fontsize = kwargs.pop('fontsize', 12)
    x_min = kwargs.pop('x_min', min(min(x1), min(x2)))
    x_max = kwargs.pop('x_max', max(max(x1), max(x2)))
    filename = kwargs.pop('filename', 'not-save')

    fig, ax1 = plt.subplots(figsize=(1280*PX, 720*PX))


    color = color1
    ax1.set_ylabel(y1_label, color=color, fontsize = fontsize)
    ax1.plot(x1, y1, color=color, label=y1_legend, linewidth=0.50)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = color2
    ax2.set_ylabel(y2_label, color=color, fontsize = 12)
    ax2.plot(x2, y2, color=color, label=y2_legend, linewidth=0.50)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xlabel(x_label)
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

    ax = plt.gca()
    ax.set_xlim([x_min, x_max])
    ax.set_title(title)

    plt.grid(alpha=0.25)
    fig.tight_layout()

    # save fig if have filename
    if filename != 'not-save':
        plt.savefig(os.path.join(IMAGES_FOLDER, f'{filename}.png'))
    plt.show()

    plt.show()