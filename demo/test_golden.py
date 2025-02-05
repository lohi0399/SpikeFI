import os
import torch
import matplotlib.pyplot as plt
import slayerSNN as snn
import demo as cs

# ✅ Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ Load trained model
fnetname = cs.get_fnetname(trial=2)  # Change trial if needed
model_path = os.path.join(cs.OUT_DIR,cs.CASE_STUDY, fnetname)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Loading trained model from: {model_path}")
net = torch.load(model_path, map_location=device)
net.eval()

# ✅ Load test dataset
test_loader = cs.test_loader
print(f"Test set size: {len(test_loader.dataset)} images")

# ✅ Compute accuracy on the full test set
correct, total = 0, 0

for _, input, target, label in test_loader:
    input = input.to(device)

    # ✅ Ensure input shape is correct (Fixing conv3d error)
    input = input.squeeze()  # Remove unnecessary dimensions
    if len(input.shape) == 6:  # If still incorrect, reshape manually
        input = input.view(1, 12, 2, 34, 34)

    # ✅ Run forward pass
    output = net.forward(input)
    predicted = snn.predict.getClass(output)

    correct += (predicted == label).sum().item()
    total += label.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
