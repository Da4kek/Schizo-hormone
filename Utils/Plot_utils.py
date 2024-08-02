import mne 
import matplotlib.pyplot as plt 
import numpy as np 

def plot_correlation_matrix(corr_matrix,title,channel_names):
    """
    Plot the correlation matrix with given channel names.

    Parameters:
    corr_matrix (numpy.ndarray): Correlation matrix to be plotted.
    title (str): Title of the plot.
    channel_names (list): List of channel names.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.xticks(ticks=np.arange(len(channel_names)),
               labels=channel_names, rotation=90)
    plt.yticks(ticks=np.arange(len(channel_names)), labels=channel_names)
    plt.tight_layout()
    plt.show()


def plot_topomap(corr_diff, ch_names, pos, title):
    """
    Plot a topographic map of the correlation differences.

    Parameters:
    corr_diff (numpy.ndarray): Correlation differences to be plotted.
    ch_names (list): List of channel names.
    pos (numpy.ndarray): 2D array of channel positions.
    title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    mne.viz.plot_topomap(corr_diff, pos, names=ch_names,
                        axes=ax, cmap='coolwarm', show=False)
    ax.set_title(title)
    plt.show()


def plot_topomap_with_regions(corr_diff, region_names, pos, title):
    """
    Plot a topographic map with anatomical region names.

    Parameters:
    corr_diff (numpy.ndarray): Correlation differences to be plotted.
    region_names (list): List of anatomical region names corresponding to the channels.
    pos (numpy.ndarray): 2D array of channel positions.
    title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    mne.viz.plot_topomap(corr_diff, pos, names=region_names,
                        axes=ax, cmap='coolwarm', show=False)
    ax.set_title(title)
    plt.show()


def plot_raster(data, time, title, event_times=None, ch_names=None):
    """
    Generate a raster plot for the given EEG data.

    Parameters:
    data (numpy.ndarray): The EEG data matrix (channels x timepoints).
    time (numpy.ndarray): Array of time points corresponding to the data columns.
    title (str): Title of the plot.
    event_times (list): Optional; List of event times to mark on the raster plot.
    ch_names (list): Optional; List of channel names.
    """
    plt.figure(figsize=(10, 8))
    for i in range(data.shape[0]):
        spike_times = time[np.where(data[i, :] > 0)[0]]
        plt.scatter(spike_times, np.ones_like(
            spike_times) * i, s=1, color='black')
    if event_times is not None:
        for event in event_times:
            plt.axvline(x=event, color='red', linestyle='--')
    plt.yticks(ticks=np.arange(
        data.shape[0]), labels=ch_names if ch_names else np.arange(data.shape[0]))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Channels')
    plt.title(title)
    plt.tight_layout()
    plt.show()
