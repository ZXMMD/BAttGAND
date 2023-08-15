import os
import torch
from torchvision.utils import save_image
from PIL import ImageFile
from model import ConvolutionalAutoencoder
import torch.optim as optim
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from loss import contrastive_loss
from dataset import get_train_loader


def train_pipeline():
    # PARAMS
    batch_size = 32
    epochs = 300
    learning_rate = 0.0001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, train_num = get_train_loader(data_dir='./data', batch_size=batch_size,
                                               num_workers=0, shuffle=True, img_size=224)

    print("using {} images for training.".format(train_num))
    train_steps = len(train_loader)

    # GPU
    net = ConvolutionalAutoencoder().to(device)
    net.load_state_dict(torch.load('./weights5/model_epoch_030.pth'))
    
    if os.path.exists("./weights6") is False:
        os.makedirs("./weights6")

    # Loss & Optimizer
    loss_func = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.00005)
    
    for epoch in range(epochs):

        # Train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        i = 0

        for step, data in enumerate(train_bar):
            i = i + 1
            input = data[0].to(device)
            adversarial = data[1].to(device)

            optimizer.zero_grad()

            output = net(input)
            loss = loss_func(output, adversarial)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_bar.desc = "epoch[{}/{}] loss:{:.8f}".format(epoch + 1,
                                                               epochs,
                                                               loss)
            if i % 100 == 0:
                save_image(input, './images_in/epoch_{:0>3d}_in_{:0>4d}.png'.format(epoch + 1, i))
                save_image(output, './images_out/epoch_{:0>3d}_out_{:0>4d}.png'.format(epoch + 1, i))
                save_image(adversarial, './images_target/epoch_{:0>3d}_target_{:0>4d}.png'.format(epoch + 1, i))
        print('epoch [{}/{}], loss: {:.8f}'.format(epoch + 1, epochs, running_loss / train_steps))

        
        torch.save(net.state_dict(), "./weights/model_epoch_{:0>3d}.pth".format(epoch + 1))


if __name__ == "__main__":
    train_pipeline()
