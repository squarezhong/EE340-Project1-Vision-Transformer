import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# get confusion matrix and plot the heatmap
def plot_confusion_matrix(cm, state):
    # use confusion matrix to show the prediction result directly
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.xlabel('predictions')
    plt.ylabel('labels')
    plt.title('Confusion Matrix for {}'.format(state))
    # automatically save the figures
    plt.savefig('confusion_matrix_{}.png'.format(state), dpi=600)
    plt.show()

def get_classification_report(true, pred, state):
    report = classification_report(true, pred, output_dict=True)
    # print(report)
    tmp = pd.DataFrame(report).transpose()
    tmp.to_csv('classification_report_{}.csv'.format(state), index=True)

    return report

# plot loss curve for a training progress
def plot_loss_curve(loss_list) -> None:
    # plot the trend of loss
    plt.plot(loss_list)
    plt.title('Loss versus Epochs for ViT')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    # save the figures
    plt.savefig('loss_figure.png', dpi=600)
    plt.show()