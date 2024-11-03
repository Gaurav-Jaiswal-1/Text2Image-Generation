import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.CGAN import *
from utils.utils import *
from models.CGAN import Discriminator



mnist_shape = (1, 28, 28)
n_classes = 10

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
device  = 'cuda'

transforms = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
  ]
)

dataloader = DataLoader(
  MNIST(
    r"C:\Users\Gaurav\OneDrive\Desktop\Text2Image\data\processed", download=False, transform=transforms

  )
)

generator_input_dim, discrminator_image_channels = calculate_input_dim(z_dim, mnist_shape, n_classes)




gen = Generator(input_dim = generator_input_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)
disc = Discriminator(image_channel = discrminator_image_channels).to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr = lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

cur_step = 0
generator_losses = []
discriminator_losses = []

noise_and_labels = False
fake = False

fake_images_and_labels = False
real_images_and_labels = False
disc_fake_pred = False
dis_real_pred = False 

for epoch in range(n_epochs):
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        one_hot_labels = one_hot_encode_vector_from_labels(labels.to(device), n_classes)
        print("one_hot_labels.size", one_hot_labels.size)


        image_one_hot_labels = image_one_hot_labels[:, :, None, None]
        print('image_one_hot_labels.size', image_one_hot_labels.size())

        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

        print("image_one_hot_labels.size", image_one_hot_labels.size())




        disc_opt.zero_grad()
        fake_noise = create_noise_vector(cur_batch_size, z_dim, device=device) 


        noise_and_labels = concat_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)

        assert len(fake) == len(real)


        fake_images_and_labels = concat_vectors(fake, one_hot_labels)
        real_images_and_labels = concat_vectors(real, one_hot_labels)

        disc_real_pred = disc(real_images_and_labels)
        disc_fake_pred = disc(fake_images_and_labels)

        assert len(disc_real_pred) == len(real)
        assert torch.any(fake_images_and_labels != real_images_and_labels)

        disc_real_loss = criterion(dis_real_pred, torch.ones_like(dis_real_pred))
        disc_fake_loss = criterion(dis_fake_pred, torch.zeros_like(dis_fake_pred))
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        dis_fake_pred = disc(fake_images_and_labels)


        discriminator_losses += [disc_loss.item()]



        # Generator training

        gen_opt.zero_grad()
        fake_images_and_labels = concat_vectors(fake, one_hot_labels)
        disc_fake_pred = disc(fake_images_and_labels)

        gen_loss = criterion(disc_fake_pred, torch.ones_likes(disc_fake_pred))

        gen_loss.backward()
        gen_opt.step()
        generator_losses += [gen_loss.item()] 




        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(f"Generator loss after {cur_step} Discriminator: {disc_mean} steps: {gen_mean}")


            plot_images_from_tensor(fake)
            plot_images_from_tensor(real)

            step_bins = 20
            x_axis = sorted(
                [i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins
            )

            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples])
                .view(-1, step_bins)
                .mean(1),
                label = "Generator Loss",


            )

            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(discriminator_losses[:num_examples])
                .view(-1, step_bins)
                .mean(1),
                label = "Discriminator Loss",
            )

            plt.legend()
            plt.show()
        elif cur_step == 0:
            print("Let long training continue")

        cur_step += 1