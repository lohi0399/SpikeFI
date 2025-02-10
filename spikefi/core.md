### **📌 Understanding How the Model Weights Are Extracted and Modified in `core.py` (SpikeFI)**  

The `core.py` file defines the **Campaign-based fault injection system** in SpikeFI. This system allows injecting faults into Spiking Neural Networks (SNNs) and analyzing their impact.  

Since you specifically **want to see where model weights are extracted and modified**, I will focus on **where and how the weights are accessed, perturbed (modified), and restored**.  

---

## **🔹 High-Level Overview of the Code**
The `Campaign` class is the main controller that:  
1. **Loads a neural network (`self.golden`)** and stores it for evaluation.  
2. **Creates a faulty version (`self.faulty`)** where faults are injected.  
3. **Injects faults into neurons, synapses (weights), and parameters** in structured rounds.  
4. **Tracks performance metrics** to evaluate the impact of faults.  

The **modification of model weights (kernels in CNNs) happens inside these functions**:  
1. **`_synaptic_pre_hook_wrapper()`** → Modifies weights before the layer executes.  
2. **`_synaptic_hook_wrapper()`** → Restores weights after execution.  
3. **`_perturb_net()`** → Registers the hooks that modify weights.  
4. **`_evaluate_train()`** → Applies faults while training.  
5. **`_evaluate_single()`, `_evaluate_O0()`, `_evaluate_O1()`, `_evaluate_optimized()`** → Apply faults during inference.  

---

## **🔹 Step 1: Extracting Model Weights**
📌 **Where are the model weights stored?**  

- The model is passed as `net` when `Campaign` is created:
  ```python
  self.golden = net  # The original "golden" model (fault-free)
  ```
- When faults are injected, a **deep copy** of the model is created:
  ```python
  self.faulty = deepcopy(self.golden)  # Faulty copy of the model
  ```
  **This ensures that the original model remains unchanged**, while `self.faulty` is modified.

---

## **🔹 Step 2: Registering Hooks to Modify Weights**
📌 **How are weights modified?**  
- **Hooks are registered** to modify weights **before and after forward passes**.

### **✅ 1. `_perturb_net()`**
- This function **registers weight modification hooks** into layers that have faults:
  ```python
  def _perturb_net(self, round: ff.FaultRound) -> None:
      ind_syn = ff.FaultTarget.W.get_index()  # Index for synaptic faults

      for layer_name in self.layers_info.get_injectables():
          self.handles.setdefault(layer_name, [[None] * 2 for _ in range(3)])
          layer = getattr(self.faulty, layer_name)

          # Register synaptic faults
          if round.any_synaptic(layer_name) and not any(self.handles[layer_name][ind_syn]):
              pre_hook = self._synaptic_pre_hook_wrapper(layer_name)
              self.handles[layer_name][ind_syn][0] = layer.register_forward_pre_hook(pre_hook)

              hook = self._synaptic_hook_wrapper(layer_name)
              self.handles[layer_name][ind_syn][1] = layer.register_forward_hook(hook)
  ```
  ✅ This **attaches hooks** (`_synaptic_pre_hook_wrapper` and `_synaptic_hook_wrapper`) to the **faulty model’s layers**.

---

### **✅ 2. `_synaptic_pre_hook_wrapper()` (Modify Weights Before Execution)**
This function **modifies the model’s weights before the layer runs**.

```python
def _synaptic_pre_hook_wrapper(self, layer_name: str, faults: list[ff.Fault] = None) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
    def synaptic_pre_hook(layer: nn.Module, _) -> None:
        for fault in faults or self.orounds[self.r_idx].search_synaptic(layer_name):
            for site in fault.sites:
                ind = site.unroll()  # Get indices of the weights to be modified
                with torch.no_grad():
                    layer.weight[ind] = fault.model.perturb_store(layer.weight[ind].cpu(), site)
    return synaptic_pre_hook
```

🔹 **What happens here?**
- It **loops over faults** in the given layer.
- It **modifies the layer’s weights (`layer.weight`) at specific indices (`site.unroll()`)**.
- It **stores the perturbed values using `fault.model.perturb_store()`**.
- The `torch.no_grad()` **prevents gradients from updating these changes**, meaning the faults persist.

---

### **✅ 3. `_synaptic_hook_wrapper()` (Restore Weights After Execution)**
This function **restores the weights back to their original values** after execution.

```python
def _synaptic_hook_wrapper(self, layer_name: str) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
    def synaptic_hook(layer: nn.Module, _, __) -> None:
        for fault in self.orounds[self.r_idx].search_synaptic(layer_name):
            for site in fault.sites:
                with torch.no_grad():
                    layer.weight[site.unroll()] = fault.model.restore(site)
    return synaptic_hook
```
🔹 **What happens here?**
- It **searches for faults in the given layer**.
- It **restores weights (`layer.weight`) to their original values** using `fault.model.restore()`.
- This ensures the faulty weights are **only active during the forward pass** and reset afterward.

---

## **🔹 Step 3: Applying Faults During Training**
📌 **Where are the faults actually applied?**  
- During training, `_evaluate_train()` runs forward passes with faults injected.

```python
def _evaluate_train(self, faulty: nn.Module, epochs: int, train_loader: DataLoader, optimizer: Optimizer, spike_loss: snn.loss) -> None:
    for _, (_, input, target, label) in enumerate(train_loader):
        output = faulty.forward(input.to(self.device))

        loss = spike_loss.numSpikes(output, target)
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()
```
🔹 **What happens here?**
- **Faults are injected through hooks** before `forward()`.
- The model runs forward, **applying weight perturbations**.
- The optimizer updates weights **with the faults included**.
- The weights **retain faults unless `_synaptic_hook_wrapper` restores them**.

---

## **🔹 Step 4: Applying Faults During Evaluation**
📌 **Where are the faults applied during inference?**  
- The same hooks (`_synaptic_pre_hook_wrapper`) apply faults in:
  - `_evaluate_single()`
  - `_evaluate_O0()`
  - `_evaluate_O1()`
  - `_evaluate_optimized()`
  
Example:
```python
def _evaluate_single(self, test_loader: DataLoader, spike_loss: snn.loss = None) -> None:
    for _, (_, input, target, label) in enumerate(test_loader):
        output = self.faulty(input.to(self.device))
```
🔹 **Faults are injected into the faulty model before each forward pass**.

---

## **🔹 Key Takeaways**
| 🔍 Step | ✅ Action |
|---------|----------|
| **1. Extract Model Weights** | `self.faulty = deepcopy(self.golden)` (Creates a faulty copy of the model) |
| **2. Register Hooks** | `_perturb_net()` attaches `_synaptic_pre_hook_wrapper()` |
| **3. Modify Weights Before Execution** | `_synaptic_pre_hook_wrapper()` perturbs `layer.weight` |
| **4. Restore Weights After Execution** | `_synaptic_hook_wrapper()` resets `layer.weight` |
| **5. Apply Faults During Training** | `_evaluate_train()` runs forward/backward with perturbed weights |
| **6. Apply Faults During Evaluation** | `_evaluate_single()` and others inject faults into test inference |

---

## **🚀 Final Answer:**
✅ **The model weights (kernels) are extracted from `self.faulty`, modified in `_synaptic_pre_hook_wrapper()`, and restored in `_synaptic_hook_wrapper()`.**  
✅ **These modifications happen dynamically during forward passes using PyTorch hooks.**  