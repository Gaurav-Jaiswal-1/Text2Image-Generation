import torch
import torch.nn as nn
from torchvision import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F


torch.manual_seed(0)


def plot_images_from_tensor(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
  image_tensor = (image_tensor + 1) / 2 
  image_unflat = image_tensor.detach().cpu()
  image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
  plt.imshow(image_grid.permute(1, 2, 0).squeeze())
  if show:
    plt.show()



def weights_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    torch.nn.init.normal_(m.weight, 0.0, 0.02)
  if isinstance(m, nn.BatchNorm2d):
    torch.nn.init.normal_(m.weight, 0.0, 0.02)
    torch.nn.init.constant_(m.bias, 0)

def one_hot_encode_vector_from_labels(labels, n_classes):
  return F.one_hot(labels, num_classes=n_classes)

def concat_vectors(x, y):
  combined = torch.cat((x.float(), y.float()), 1)
  return combined

def calculate_input_dim(z_dim, mnist_shape, n_classes):
  generator_input_dim = z_dim + n_classes
  disciminator_image_channels = mnist_shape[0] + n_classes
  return generator_input_dim, disciminator_image_channels