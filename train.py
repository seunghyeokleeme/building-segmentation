import os
import torch
from torch import nn
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#parameters
lr = 1e-3
batch_size = 2
num_epoch = 100

data_dir = './datasets'
log_dir = './log'
ckpt_dir = './checkpoint'

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # mac
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using {device} device")

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'exp1/train'), comment='train')
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'exp1/val'), comment='val')

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)

        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)

        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)

        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.maxpool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.maxpool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.maxpool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.maxpool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


class xbdDataset(VisionDataset):
    def __init__(self, root = None, transforms = None, transform = None, target_transform = None):
        super(xbdDataset, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "targets")
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))
        assert len(self.image_filenames) == len(self.mask_filenames)

    def __len__(self):
        return len(self.mask_filenames)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[index])

        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

        return image, mask

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.3096, 0.3428, 0.2564],
                         std=[0.1309, 0.1144, 0.1081])])
target_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

train_data = xbdDataset(root=os.path.join(data_dir, 'train'), transform=transform, target_transform=target_transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True)

val_data = xbdDataset(root=os.path.join(data_dir, 'hold'), transform=transform, target_transform=target_transform)
val_loader = DataLoader(val_data, batch_size, shuffle=False)

test_data = xbdDataset(root=os.path.join(data_dir, 'test'), transform=transform, target_transform=target_transform)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

net = UNet().to(device)

# Loss function 설정하기
fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_pred = lambda output: (torch.sigmoid(output) > 0.5).float()
fn_denorm = lambda x, mean, std: x * torch.tensor(std, device=x.device).view(1, -1, 1, 1) + torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
fn_acc = lambda pred, label: (pred == label).float().mean()

# Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))



def train_loop(dataloader, model, fn_loss, optim, epoch, writer):
    num_batches = len(dataloader)
    train_loss, correct = 0.0, 0

    model.train()

    for batch, (input, mask) in enumerate(dataloader, 1):
        input = input.to(device)
        mask = mask.to(device)
        
        logits = model(input)
        loss = fn_loss(logits, mask)

        # backward pass
        loss.backward()
        optim.step()
        optim.zero_grad()

        train_loss += loss.item()
        
        pred = fn_pred(logits)
        acc = fn_acc(pred, mask)
        correct += acc.item()

        print("TRAIN: EPOCH %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
                (epoch, batch, num_batches, loss.item(), acc.item()))

        input_img = fn_denorm(input, mean=[0.3096, 0.3428, 0.2564], std=[0.1309, 0.1144, 0.1081])

        # Tensorboard 저장하기
        writer.add_images("input", img_tensor=input_img, global_step=(num_batches*(epoch -1)+batch))
        writer.add_images("mask", img_tensor=mask, global_step=(num_batches*(epoch -1)+batch))
        writer.add_images("predict", img_tensor=pred, global_step=(num_batches*(epoch -1)+batch))
    
    train_loss /= num_batches
    train_accuracy = correct / num_batches
    writer.add_scalar('Loss', train_loss, epoch)
    writer.add_scalar('Accuracy', train_accuracy, epoch)
    
    return train_loss, train_accuracy

def eval_loop(dataloader, model, fn_loss, epoch, writer):
    num_batches = len(dataloader)
    eval_loss, correct = 0.0, 0

    model.eval()

    with torch.no_grad():
        for batch, (input, mask) in enumerate(dataloader, 1):
            input = input.to(device)
            mask = mask.to(device)
            
            logits = model(input)
            loss = fn_loss(logits, mask)

            eval_loss += loss.item()
            
            pred = fn_pred(logits)
            acc = fn_acc(pred, mask)
            correct += acc.item()

            print("VALID: EPOCH %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
                    (epoch, batch, num_batches, loss.item(), acc.item()))

            input_img = fn_denorm(input, mean=[0.3096, 0.3428, 0.2564], std=[0.1309, 0.1144, 0.1081])

            # Tensorboard 저장하기
            writer.add_images("input", img_tensor=input_img, global_step=(num_batches*(epoch -1)+batch))
            writer.add_images("mask", img_tensor=mask, global_step=(num_batches*(epoch -1)+batch))
            writer.add_images("predict", img_tensor=pred, global_step=(num_batches*(epoch -1)+batch))
        
    eval_loss /= num_batches
    eval_accuracy = correct / num_batches
    writer.add_scalar('Loss', eval_loss, epoch)
    writer.add_scalar('Accuracy', eval_accuracy, epoch)
    
    return eval_loss, eval_accuracy

for epoch in range(1, num_epoch+1):
    loss, acc = train_loop(dataloader=train_loader, model=net, fn_loss=fn_loss, optim=optim, epoch=epoch, writer=writer_train)
    val_loss, val_acc = eval_loop(dataloader=val_loader, model=net, fn_loss=fn_loss, epoch=epoch, writer=writer_val)

    print(f"Epoch {epoch} summary: Train Loss: {loss:.4f} | Train Accuracy: {100*acc:.1f}% | Val Loss: {val_loss:.4f} | Val Accuracy: {100*val_acc:.1f}%")
    if epoch % 10 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()