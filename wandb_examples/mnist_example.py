#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from models import SimpleClassifier

# from wandb import


#%%
wandb.init(project="mnist-example",
           config={"batch_size": 2000,
                   "test_batch_size": 1000,
                   "epochs":10,
                   "lr": 0.001,
                   "weight_decay": 0.01,
                   "no_cuda": False,
                   "seed": 42,
                   "log_interval": 100}
           )

#%%
config=wandb.config

# Set random seed for reproducibility
torch.manual_seed(config.seed)

# Check if CUDA is available
use_cuda = not config.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset - use only train split and divide it 80/20
full_train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)

# Split the training dataset into train (80%) and test (20%)
train_size = int(0.8 * len(full_train_dataset))
test_size = len(full_train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

dataset_artifact=wandb.Artifact(name="MNIST",type="Datasets")
wandb.run.log_artifact(artifact_or_path=dataset_artifact)
# Log dataset examples to wandb
example_images = []
example_labels = []
for i in range(10):  # Get one example of each digit
    image, label = train_dataset[i]
    example_images.append(image.numpy())
    example_labels.append(label)

# Create a grid of images
example_images = np.array(example_images)
example_images = np.transpose(example_images, (0, 2, 3, 1))  # NCHW -> NHWC
example_images = example_images.squeeze()

#%%
wandb.log({
    "examples": [wandb.Image(img, caption=f"Label: {label}")
                 for img, label in zip(example_images, example_labels)]
})

#%%
# Define the CNN model


# Initialize the model, loss function, and optimizer
model = SimpleClassifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss()

# Log model architecture to wandb
wandb.watch(model, log="all")


# Training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % config.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": loss.item(),
                "batch": batch_idx
            })


# Testing function
def test(epoch,last_epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                class_pred = output.argmax(dim=1)
                all_preds.extend(class_pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    # Log test metrics to wandb
    wandb.log({
        "epoch": epoch,
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "f1_score": f1_score(all_targets,all_preds,average="macro"),
        "precision":precision_score(all_targets,all_preds,average="macro"),
        "recall":recall_score(all_targets,all_preds,average="macro")
    })

    # Log confusion matrix


    if epoch==last_epoch:
        wandb.log({
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_targets,
                preds=all_preds,
                class_names=[str(i) for i in range(10)]
            )
        })


#%%
for epoch in range(1, config.epochs + 1):

    train(epoch)
    test(epoch,config.epochs)

# Save the model
model_name=f"mnist_cnn_{wandb.run.name}"
model_artifact = wandb.Artifact(name=model_name, type="model")

torch.save(model.state_dict(), rf"save-models\{model_name}.pt")

model_artifact.add_file(rf"save-models\{model_name}.pt")
wandb.run.log_artifact(artifact_or_path=model_artifact)
#%%
# wandb.save("mnist_cnn.pt")

# Finish the wandb run
wandb.finish()

print("Training completed! Model saved as mnist_cnn.pt")