import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pywt
import numpy as np
from PIL import Image

class DWT2(object):

    def __init__(self, wavelet):
        self.wavelet = wavelet

    def __call__(self, sample):

        sample_array = np.array(sample)
        sample_r = sample_array[:, :, 0]
        sample_g = sample_array[:, :, 1]
        sample_b = sample_array[:, :, 2]

        coeffs2_r = pywt.dwt2(sample_r, self.wavelet)
        cA_r, (cH_r, cV_r, cD_r) = coeffs2_r
        wave_param_r = np.vstack((np.hstack((cA_r, cH_r)), np.hstack((cV_r, cD_r))))

        coeffs2_g = pywt.dwt2(sample_g, self.wavelet)
        cA_g, (cH_g, cV_g, cD_g) = coeffs2_g
        wave_param_g = np.vstack((np.hstack((cA_g, cH_g)), np.hstack((cV_g, cD_g))))

        coeffs2_b = pywt.dwt2(sample_b, self.wavelet)
        cA_b, (cH_b, cV_b, cD_b) = coeffs2_b
        wave_param_b = np.vstack((np.hstack((cA_b, cH_b)), np.hstack((cV_b, cD_b))))

        wave_param = np.array([wave_param_r, wave_param_g, wave_param_b])
        wave_param_img = np.transpose(wave_param, (1, 2, 0))

        sample.show()
        img = Image.fromarray(wave_param_img.astype(np.uint8), mode='RGB')
        img.show()

        import pdb; pdb.set_trace()
        return np.float32(wave_param)
        # return sample


data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    DWT2('haar'),
    transforms.ToTensor()
    ])

train_data = datasets.GTSRB(root = 'data', download=True, split='train', transform = data_transforms)

test_data = datasets.GTSRB(root = 'data', download=True, split='test', transform=data_transforms)

# Create data loader for training and validation
BATCH_SIZE = 256

train_loader = DataLoader(train_data, shuffle=True, batch_size = BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size = BATCH_SIZE)

# https://mailto-surajk.medium.com/a-tutorial-on-traffic-sign-classification-using-pytorch-dabc428909d7

class AlexnetTS(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*7*7, 1000),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, output_dim)
            )
        
    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

numClasses = 43

model = AlexnetTS(numClasses)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

learning_rate = 0.001
EPOCHS = 50

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    # Train the model
    model.train()
    
    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Training pass
        opt.zero_grad()
        
        output, _ = model(images)
        loss = criterion(output, labels)
        
        # Backpropagation
        loss.backward()
        
        # Calculate accuracy
        acc = calculate_accuracy(output, labels)
        
        # Optimizing weights
        opt.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)
    
    
# Function to perform evaluation on the trained model
def evaluate(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    # Evaluate the model
    model.eval()
    
    with torch.no_grad():
        for (images, labels) in loader:
            images = images.cuda()
            labels = labels.cuda()
            
            # Run predictions
            output, _ = model(images)
            loss = criterion(output, labels)
            
            # Calculate accuracy
            acc = calculate_accuracy(output, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc   
 
# Perform training
# List to save training and val loss and accuracies
train_loss_list = [0]*EPOCHS
train_acc_list = [0]*EPOCHS
val_loss_list = [0]*EPOCHS
val_acc_list = [0]*EPOCHS

for epoch in range(EPOCHS):
    print("Epoch-%d: " % (epoch))

    train_loss, train_acc = train(model, train_loader, optimizer, criterion)

    val_loss, val_acc = evaluate(model, test_loader, optimizer, criterion)

    
    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc
    
    print("Training: Loss = %.4f, Accuracy = %.4f" % (train_loss, train_acc))
    print("Validation: Loss = %.4f, Accuracy = %.4f" % (val_loss, val_acc))
    print("")