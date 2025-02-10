### **📌 Understanding the Code in `__main__.py` That Saves the Model as a `.pkl` File**  

The snippet you provided:  
```python
elif CMPN_SEL == 2:
    cmpn2 = sfi.Campaign(net, cs.shape_in, net.slayer,
                         name=fnetname.removesuffix('.pt') + '_test')

    cmpn2.inject_complete(sfi.fm.DeadNeuron(), layer_names=['SF2'])

    print(cmpn2.name)
    cmpn2.run(cs.test_loader, spike_loss=snn.loss(cs.net_params).to(cmpn2.device))

    cmpn2.save()
    print(f"{cmpn2.duration : .2f} secs")
```
This code **creates, runs, and saves a fault injection campaign** using **SpikeFI**.  
Let's break it down **step by step**.

---

## **🔹 Step 1: Creating a Fault Injection Campaign**
```python
cmpn2 = sfi.Campaign(net, cs.shape_in, net.slayer,
                     name=fnetname.removesuffix('.pt') + '_test')
```
- **Creates a new campaign (`cmpn2`)** for fault injection.  
- The `Campaign` class manages fault injection experiments.  
- It initializes with:
  - `net` → The **original SNN model**.
  - `cs.shape_in` → The **input shape**.
  - `net.slayer` → The **SLAYER framework** (used for SNNs).
  - `name=fnetname.removesuffix('.pt') + '_test'` → Generates a name **based on the original model name**.

✅ **At this point,** `cmpn2` holds a **fault-free copy** of the model (`self.golden`).  
The **faulty model (`self.faulty`) will be created later** when faults are injected.

---

## **🔹 Step 2: Injecting a Fault (`DeadNeuron`)**
```python
cmpn2.inject_complete(sfi.fm.DeadNeuron(), layer_names=['SF2'])
```
- Injects a **DeadNeuron** fault into `SF2` (the second fully connected layer).
- This means **some neurons in `SF2` will be permanently inactive (output = 0)**.

✅ **Now, `cmpn2.faulty` contains a faulty version of the model.**  
The `SF2` layer will have **dead neurons** when the model runs.

---

## **🔹 Step 3: Running the Faulty Model**
```python
cmpn2.run(cs.test_loader, spike_loss=snn.loss(cs.net_params).to(cmpn2.device))
```
- Runs **inference** on the faulty model (`cmpn2.faulty`).
- Uses `cs.test_loader` (test dataset) to evaluate performance.
- Uses **spike-based loss (`snn.loss()`)**.

✅ **This executes the faulty model** to see how `DeadNeuron` faults affect accuracy.

---

## **🔹 Step 4: Saving the Faulty Model (`.pkl` File)**
```python
cmpn2.save()
```
- Saves `cmpn2` as a **`.pkl` (pickle) file**.
- The `.pkl` file contains:
  - The **fault injection settings**.
  - The **faulty network state** (`cmpn2.faulty`).
  - The **performance statistics**.
  - The **fault injection rounds**.

✅ **The `.pkl` file does NOT store raw model weights**.  
Instead, it stores a **representation of the campaign**, including:
1. The **original model (`self.golden`)**.
2. The **faulty model (`self.faulty`)**.
3. The **fault injection details**.

---

## **🔹 Step 5: Printing Execution Time**
```python
print(f"{cmpn2.duration : .2f} secs")
```
- Displays the **total time taken for the experiment**.

---

## **🔹 Does the `.pkl` File Contain Model Weights?**
Not directly. The `.pkl` file **does not store raw model weights**, but it does contain:
- The **original model (`self.golden`)**.
- The **fault-injected model (`self.faulty`)**.
- The **fault configurations**.

✅ **You can extract the faulty weights from the `.pkl` file by loading it.**  
To visualize and compare **faulty vs. original weights**, follow the next steps.

---

## **📌 How to Load and Compare Faulty vs. Original Weights**
1. **Load the `.pkl` file**
```python
import torch
import pickle

with open("faulty_model.pkl", "rb") as file:
    cmpn2 = pickle.load(file)  # Load the Campaign object
```
2. **Extract Original and Faulty Model Weights**
```python
golden_weights = cmpn2.golden.state_dict()  # Original model
faulty_weights = cmpn2.faulty.state_dict()  # Fault-injected model
```
3. **Compare Weights for `SF2`**
```python
import matplotlib.pyplot as plt

sf2_layer = "SF2.weight"

golden_w = golden_weights[sf2_layer].cpu().numpy()
faulty_w = faulty_weights[sf2_layer].cpu().numpy()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(golden_w, cmap='viridis')
plt.title("Original Weights")

plt.subplot(1,2,2)
plt.imshow(faulty_w, cmap='viridis')
plt.title("Faulty Weights (DeadNeuron)")

plt.show()
```
✅ **This will visualize `SF2` weights before and after fault injection**.

---

## **🔹 Summary**
| Step | Action |
|------|--------|
| **1. Create Campaign** | Initializes `cmpn2` with the original model. |
| **2. Inject Fault** | Applies `DeadNeuron` faults to `SF2`. |
| **3. Run Experiment** | Evaluates model performance with faults. |
| **4. Save Campaign (`.pkl`)** | Stores fault settings & model states. |
| **5. Load `.pkl` and Compare Weights** | Extract & visualize weights before/after faults. |

✅ **The `.pkl` file does not directly store weights, but contains the faulty model, which includes modified weights.**  
✅ **You can extract weights using `state_dict()` and visualize changes.**  
