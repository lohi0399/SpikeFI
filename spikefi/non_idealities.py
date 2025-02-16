# Memristor non-ideality models.
# This file extends fault.py and models.py to model memristor-specific faults. I basically use inheritance to define new non-idealitity models.
# Author: Lohit Gandham

import torch
from torch import Tensor
import random
from spikefi.fault import FaultModel, FaultSite, FaultTarget
import pandas as pd
import matplotlib.pyplot as plt

class VariabilityFault(FaultModel):
    """
    Models Device-to-Device (D2D) and Cycle-to-Cycle (C2C) variability
    as a stochastic weight perturbation.
    """
    def __init__(self, variability_factor: float):
        super().__init__(FaultTarget.WEIGHT, self.perturb_variability, variability_factor)

    def perturb_variability(self, original_weight: Tensor, factor: float) -> Tensor:
        noise = factor * (2 * torch.rand_like(original_weight) - 1)  # Uniform noise [-factor, factor]
        return original_weight * (1 + noise)


class IRDropFault(FaultModel):
    """
    Models IR drop in memristor crossbars by reducing weights
    based on their position in the crossbar (closer to edges = more IR drop).
    """
    def __init__(self, max_drop: float, crossbar_size: tuple[int, int]):
        super().__init__(FaultTarget.WEIGHT, self.perturb_ir_drop, max_drop, crossbar_size) #  super().__init__() calls the constructor of the parent class

    def perturb_ir_drop(self, original_weight: Tensor, max_drop: float, size: tuple[int, int]) -> Tensor:
        rows, cols = size
        position_matrix = torch.linspace(0, 1, steps=rows).unsqueeze(1).expand(rows, cols)  # Position-based scaling
        drop_factor = 1 - max_drop * position_matrix  # More drop at higher index
        return original_weight * drop_factor.to(original_weight.device)


class IntermittentFault(FaultModel):
    """
    Models transient stuck-at faults where neurons are randomly activated/deactivated.
    """
    def __init__(self, probability: float):
        super().__init__(FaultTarget.OUTPUT, self.perturb_intermittent, probability)

    def perturb_intermittent(self, original_activation: Tensor, probability: float) -> Tensor:
        mask = torch.rand_like(original_activation) > probability  # Randomly deactivate neurons
        return original_activation * mask


class EnduranceFailure(FaultModel):
    """
    Models endurance failure where synaptic weights degrade after repeated writes.
    """
    def __init__(self, threshold_cycles: int, degradation_factor: float):
        super().__init__(FaultTarget.WEIGHT, self.perturb_endurance, threshold_cycles, degradation_factor)
        self.cycle_count = {}  # Tracks write cycles per synapse

    def perturb_endurance(self, original_weight: Tensor, threshold: int, factor: float, site: FaultSite) -> Tensor:
        if site not in self.cycle_count:
            self.cycle_count[site] = 0

        self.cycle_count[site] += 1
        if self.cycle_count[site] >= threshold:
            return original_weight * (1 - factor)  # Decrease weight after threshold cycles
        return original_weight


class RetentionFailure(FaultModel):
    """
    Models retention failure by introducing a slow weight drift over time.
    """
    def __init__(self, drift_rate: float):
        super().__init__(FaultTarget.WEIGHT, self.perturb_retention, drift_rate)
        self.time_elapsed = 0

    def perturb_retention(self, original_weight: Tensor, drift_rate: float) -> Tensor:
        self.time_elapsed += 1
        drift = drift_rate * self.time_elapsed
        return original_weight * (1 - drift)


class ReadDisturbFault(FaultModel):
    """
    Models read disturb effect where excessive reads cause unintended weight perturbations.
    """
    def __init__(self, disturb_threshold: int, disturb_factor: float):
        super().__init__(FaultTarget.WEIGHT, self.perturb_read_disturb, disturb_threshold, disturb_factor)
        self.read_count = {}

    def perturb_read_disturb(self, original_weight: Tensor, threshold: int, factor: float, site: FaultSite) -> Tensor:
        if site not in self.read_count:
            self.read_count[site] = 0

        self.read_count[site] += 1
        if self.read_count[site] >= threshold:
            return original_weight * (1 - factor)  # Reduce weight slightly after many reads
        return original_weight




# Example usage 1

# Simulated synaptic weights (5x5 matrix for simplicity)
original_weights = torch.ones((5, 5)) * 0.5  # Initial weights set to 0.5
print("Original Weights:\n", original_weights)

# Define a sample FaultSite for tracking
sample_site = FaultSite(layer_name="synapse_layer", position=(2, 2, 0, 0))

# 1. Variability (D2D & C2C)
var_fault = VariabilityFault(variability_factor=0.1)
perturbed_weights = var_fault.perturb_variability(original_weights, 0.1)
print("\nVariability Fault Applied:\n", perturbed_weights)

# 2. IR Drop
ir_drop_fault = IRDropFault(max_drop=0.3, crossbar_size=(5, 5))
ir_dropped_weights = ir_drop_fault.perturb_ir_drop(original_weights, 0.3, (5, 5))
print("\nIR Drop Fault Applied:\n", ir_dropped_weights)

# 3. Intermittent Variations (Random neuron activations/deactivations)
intermittent_fault = IntermittentFault(probability=0.2)
activations = torch.ones((5, 5))  # Simulated neuron activations
perturbed_activations = intermittent_fault.perturb_intermittent(activations, 0.2)
print("\nIntermittent Fault Applied (Neuron Activations):\n", perturbed_activations)

# 4. Endurance Failure (After N writes)
endurance_fault = EnduranceFailure(threshold_cycles=3, degradation_factor=0.2)
for i in range(5):  # Simulating multiple writes
    perturbed_weights = endurance_fault.perturb_endurance(original_weights, 3, 0.2, sample_site)
    print(f"\nEndurance Failure (After {i+1} Writes):\n", perturbed_weights)

# 5. Retention Failure (Weight drifts over time)
retention_fault = RetentionFailure(drift_rate=0.01)
for t in range(5):  # Simulating time progression
    perturbed_weights = retention_fault.perturb_retention(original_weights, 0.01)
    print(f"\nRetention Failure (After {t+1} Time Units):\n", perturbed_weights)

# 6. Read Disturb (After N reads)
read_disturb_fault = ReadDisturbFault(disturb_threshold=3, disturb_factor=0.05)
for r in range(5):  # Simulating multiple reads
    perturbed_weights = read_disturb_fault.perturb_read_disturb(original_weights, 3, 0.05, sample_site)
    print(f"\nRead Disturb Fault (After {r+1} Reads):\n", perturbed_weights)


# Example usage 2

matrix_size = (5, 5)

# Initialize original weights
original_weights = torch.ones(matrix_size) * 0.5

# Define a sample FaultSite
sample_site = FaultSite(layer_name="synapse_layer", position=(2, 2, 0, 0))

# Initialize fault models
faults = {
    "Variability": VariabilityFault(variability_factor=0.1),
    "IR Drop": IRDropFault(max_drop=0.3, crossbar_size=matrix_size),
    "Intermittent": IntermittentFault(probability=0.2),
    "Endurance": EnduranceFailure(threshold_cycles=3, degradation_factor=0.2),
    "Retention": RetentionFailure(drift_rate=0.01),
    "Read Disturb": ReadDisturbFault(disturb_threshold=3, disturb_factor=0.05)
}

# Number of iterations for fault progression
num_iterations = 10

# Dictionary to store tensors at each iteration
stored_tensors = {fault_name: [] for fault_name in faults.keys()}

# Apply faults and visualize changes over iterations
for fault_name, fault in faults.items():
    weights = original_weights.clone()  # Reset weights for each fault

    for t in range(num_iterations):
        if fault_name == "Variability":
            weights = fault.perturb_variability(weights, 0.1)
        elif fault_name == "IR Drop":
            weights = fault.perturb_ir_drop(weights, 0.3, matrix_size)
        elif fault_name == "Intermittent":
            activations = torch.ones_like(weights)  # Simulated activations
            weights = fault.perturb_intermittent(activations, 0.2)
        elif fault_name == "Endurance":
            weights = fault.perturb_endurance(weights, 3, 0.2, sample_site)
        elif fault_name == "Retention":
            weights = fault.perturb_retention(weights, 0.01)
        elif fault_name == "Read Disturb":
            weights = fault.perturb_read_disturb(weights, 3, 0.05, sample_site)

        # Store the tensor (cloning ensures we don't overwrite it in future iterations)
        stored_tensors[fault_name].append(weights.clone())

# Example: Access stored tensors (e.g., Variability fault, iteration 3)
example_tensor = stored_tensors["Variability"][3]
print("\nStored Tensor (Variability Fault, Iteration 3):\n", example_tensor)

import pickle

# Convert tensors to numpy arrays for more efficient storage
stored_tensors_np = {
    fault: [tensor.cpu().numpy() for tensor in tensor_list]
    for fault, tensor_list in stored_tensors.items()
}

# Save the stored tensors to a file
file_path = "spikefi/stored_tensors.pkl"
with open(file_path, "wb") as f:
    pickle.dump(stored_tensors_np, f)

# Confirm the file path
file_path
