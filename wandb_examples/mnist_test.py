import os
import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch import nn
from torch.utils.data import DataLoader
from models import SimpleClassifier

def load_model_from_artifact(artifact_name, model_class, device):
    """Load model from W&B artifact"""
    artifact = wandb.use_artifact(artifact_name, type='model')
    artifact_dir = artifact.download()
    
    # Find the .pt file in the artifact directory
    model_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    if not model_files:
        raise ValueError(f"No .pt model file found in artifact {artifact_name}")
    
    model_path = os.path.join(artifact_dir, model_files[0])
    
    # Initialize model and load state dict
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, model_path


def test_model(model, test_loader, device):
    """Test the model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'targets': all_targets
    }


def create_results_table(results, class_names):
    """Create a W&B table with detailed results"""
    per_class_report = classification_report(
        results['targets'], 
        results['predictions'], 
        target_names=class_names,
        output_dict=True
    )
    
    # Create table data
    table_data = []
    for class_name in class_names:
        if class_name in per_class_report:
            class_metrics = per_class_report[class_name]
            table_data.append([
                class_name,
                class_metrics['precision'],
                class_metrics['recall'],
                class_metrics['f1-score'],
                class_metrics['support']
            ])
    
    # Add overall metrics
    table_data.append([
        'Overall',
        results['precision'],
        results['recall'],
        results['f1_score'],
        len(results['targets'])
    ])
    
    # Create W&B table
    table = wandb.Table(
        columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        data=table_data
    )
    
    return table


# Initialize W&B
wandb.init(project="mnist-example", job_type="model_testing")
wandb.run.name = "testing-" + wandb.run.name

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model from artifact (replace with your artifact name)
artifact_name = "mnist-example/mnist_cnn_hearty-cherry-5:latest"  # Update this
model_artifact = None
try:
    model, model_path = load_model_from_artifact(artifact_name, SimpleClassifier, device)
    model_artifact = wandb.use_artifact(artifact_name, type='model')
    print(f"Model loaded from artifact: {model_path}")
    
    # Log that this artifact is being used
    wandb.log_artifact(model_artifact, aliases=["used"])
    
except Exception as e:
    print(f"Failed to load from artifact: {e}")
    print("Loading from local file instead...")
    model = SimpleClassifier().to(device)
    model.load_state_dict(torch.load("./save-models/mnist_cnn_swept-pine-15.pt", map_location=device))
    model.eval()

# Test the model
results = test_model(model, test_loader, device)

# Print results
print(f"\nTest Results:")
print(f"Test Accuracy: {results['accuracy']:.2f}%")
print(f"F1 Score: {results['f1_score']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")

# Create class names
class_names = [str(i) for i in range(10)]

# Create and log results table
results_table = create_results_table(results, class_names)

# Create results artifact linked to the model artifact
results_artifact = wandb.Artifact(
    name=f"results_table_{wandb.run.name}", 
    type="evaluation_results",
    description=f"Test results table generated from model artifact {artifact_name}"
)

# Add the results table to the artifact
with results_artifact.new_file("results_table.json", mode="w") as f:
    import json
    # Convert table data to JSON for storage
    table_data = {
        'columns': results_table.columns,
        'data': results_table.data,
        'test_accuracy': results['accuracy'],
        'f1_score': results['f1_score'],
        'precision': results['precision'],
        'recall': results['recall'],
        'model_artifact': artifact_name
    }
    json.dump(table_data, f, indent=2)

# Link the results artifact to the model artifact if it exists
if model_artifact:
    results_artifact.metadata.update({
        "source_model_artifact": artifact_name,
        # "model_run_id": model_artifact.source_run
    })

# Log the results artifact
wandb.log_artifact(results_artifact)

# Log metrics and table to W&B
wandb.log({
    'test_accuracy': results['accuracy'],
    'f1_score': results['f1_score'],
    'precision': results['precision'],
    'recall': results['recall'],
    'results_table': results_table,
    'confusion_matrix': wandb.plot.confusion_matrix(
        probs=None,
        y_true=results['targets'],
        preds=results['predictions'],
        class_names=class_names
    )
})

# Log test examples
example_images = []
for i in range(min(10, len(results['targets']))):
    image, true_label = test_dataset[i]
    pred_label = results['predictions'][i]
    
    # Convert tensor to numpy for logging
    img_array = image.numpy().squeeze()
    
    example_images.append(wandb.Image(
        img_array, 
        caption=f"True: {true_label}, Pred: {pred_label}"
    ))

wandb.log({"test_examples": example_images})

wandb.finish()
print("Testing completed! Results logged to W&B.")

