
import numpy as np
from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PlotLosses(keras.callbacks.Callback):
    """
        Keras Callback to plot the training loss and accuracy of the training and validation sets.
    """
    def __init__(self, metrics):
        self.i = 0
        self.epoch = []
        self.metrics_names = metrics
        self.metrics = {}

        for name in self.metrics_names:
            self.metrics[name] = []
            self.metrics['val_'+name] = []

        self.fig = plt.figure()
        self.logs = []
        # self.tf_version = float(tf.__version__[:3])

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.epoch.append(self.i)

        # extract the metrics from the logs
        for name in self.metrics_names:
            # get the training metric
            tr_value = logs.get(name)

            # get the validation metric
            try:
                val_value = logs.get('val_'+name)
            except:
                val_value = 0.0

            # store the metric: for f1-score we get two values one for each class. 
            # We only want the value for the positive class
            self.metrics[name].append(tr_value)
            self.metrics['val_'+name].append(val_value)

        self.i += 1
        f, axes = plt.subplots(len(self.metrics_names), 1, sharex=True, 
                               figsize=(12, 4 * len(self.metrics_names)))
        clear_output(wait=True)
        
        for name, ax in zip(self.metrics_names, axes):
            ax.plot(self.epoch, self.metrics.get(name), label=name)
            ax.plot(self.epoch, self.metrics.get('val_'+name), label="val "+name)
            ax.legend()

        axes[-1].set_xlabel("Epoch")
        plt.show()
		
		
def print_confusion_matrix(confusion_matrix, class_names, activities, 
  figsize = (12, 6), fontsize=10):
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the output figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    fig = fig = plt.gcf()
    heatmap.yaxis.set_ticklabels(activities, rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(activities, rotation=90, ha='right', fontsize=fontsize)
    plt.show()
	
def stylize_axes(ax, title):
    """
    Stylize the axes by removing ths spines and ticks. Also, set font size of papers. 

    ax: matplotlib axes
    title: string 

    """
    # removes the top and right lines from the plot rectangle
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top=False, direction='out', width=1)
    ax.yaxis.set_tick_params(right=False, direction='out', width=1)

    # Enforce the size of the title, label and tick labels
    ax.set_xlabel(ax.get_xlabel(), fontsize='large')
    ax.set_ylabel(ax.get_ylabel(), fontsize='large')

    ax.set_yticklabels(ax.get_yticklabels(), fontsize='medium')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize='medium')

    ax.set_title(title, fontsize='large')

def save_image(fig, title):
    """
    Save the figure as PNG and pdf files in the working directory.
    
    fig: matplotlib figure
    title: filename for the images
    """
    if title is not None:
        fig.savefig(title+".png", dpi=300, bbox_inches='tight', transparent=True)
        fig.savefig(title+".pdf", bbox_inches='tight')

def figure_size(fig, size):
    """Adjuest the figure size
    
    fig: matplotlib figure
    size: Tuple size in inches
     """
    fig.set_size_inches(size)
    fig.tight_layout()
    
def resadjust(ax, xres=None, yres=None):
    """
    Send in an axis and fix the resolution as desired.
    """

    if xres:
        start, stop = ax.get_xlim()
        ticks = np.arange(start, stop + xres, xres)
        ax.set_xticks(ticks)
    if yres:
        start, stop = ax.get_ylim()
        ticks = np.arange(start, stop + yres, yres)
        ax.set_yticks(ticks)