<p align="center">
    <img src="https://github.com/SpikeFI/.github/blob/main/profile/spikefi_logo.png" width="400">
</p>

# SpikeFI
### *A Fault Injection Framework for Spiking Neural Networks*

This is the main repository of the *SpikeFI* framework.

## Brief Introduction

Neuromorphic computing and spiking neural networks (SNNs) are gaining traction across various artificial intelligence (AI) tasks thanks to their potential for efficient energy usage and faster computation speed. This comparative advantage comes from mimicking the structure, function, and efficiency of the biological brain, which arguably is the most brilliant and green computing machine. As SNNs are eventually deployed on a hardware processor, the reliability of the application in light of hardware-level faults becomes a concern, especially for safety- and mission-critical applications. We propose *SpikeFI*, a fault injection framework for SNNs that can be used for automating the reliability analysis and test generation. *SpikeFI* is built upon the SLAYER PyTorch framework with fault injection experiments accelerated on a single or multiple GPUs. It has a comprehensive integrated neuron and synapse fault model library, in accordance to the literature in the domain, which is extendable by the user if needed. It supports: single and multiple faults; permanent and transient faults; specified, random layer-wise, and random network-wise fault locations; and pre-, during, and post-training fault injection. It also offers several optimization speedups and built-in functions for results visualization. *SpikeFI* is open-source.

## Publication

The article introducing *SpikeFI* has been submitted to IEEE for possible publication. A preprint version is available on HAL [here](https://hal.science/hal-04825966).

### Citation

To reference our work, please use the following citation:

> T. Spyrou, S. Hamdioui and H.-G. Stratigopoulos, "SpikeFI: A Fault Injection Framework for Spiking Neural Networks," arXiv, 2024, https://arxiv.org/abs/2412.06795

```bibtex
@misc{spyrou2024spikefi,
      title={SpikeFI: A Fault Injection Framework for Spiking Neural Networks}, 
      author={Theofilos Spyrou and Said Hamdioui and Haralampos-G. Stratigopoulos},
      year={2024},
      eprint={2412.06795},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2412.06795}, 
}
```

## Acknowledgments
This work was funded by the ANR RE-TRUSTING project under Grant No ANR-21-CE24-0015-03 and by the European Network of Excellence dAIEDGE under Grant Agreement No 101120726. The work of T. Spyrou was supported by the Sorbonne Center for Artificial Intelligence (SCAI) through Fellowship.


## Installation

### Clone the repository

`git clone git@github.com:SpikeFI/SpikeFI.git`

or

`git clone https://github.com/SpikeFI/SpikeFI.git`

### Requirements

Python 3 with the following packages:
- PyTorch
- SLAYER (more information at the [SLAYER-PyTorch repository](https://github.com/bamsumit/slayerPytorch))
- numpy
- matplotlib

SLAYER-PyTorch requires a CUDA-enabled GPU for training SNN models.

*SpikeFI* has been tested with CUDA libraries version 12.1 and GCC 13.2 on Almalinux 9.3 and Ubuntu 24.04 using NVIDIA A100 80GB and NVIDIA Quadro RTX 4000 GPUs.

## Documentation

*Under construction...*

## Getting Started

The *SpikeFI* main package contains the following modules:
- core.py
- fault.py
- models.py
- visual.py

### Running a FI campaign

*Under construction ...*
``` python

```

### More examples

The *demo* package contains the following modules demonstrating how to use *SpikeFI*:
- bitfilp.py
- optimizations.py
- parametric.py
- train_golden.py
- train.py

### Trained SNNs

The *nets* subpackage of the *demo* package contains the network classes (defined in SLAYER) for the N-MNIST and IBM's DVS128 Gesture SNNs, along with the classes to load their datasets. The paths to the dataset directories need to be indicated by the user in the .yaml configuration file of the network.

## License & Copyright

Theofilos Spyrou is the Author of the Software <br>
Copyright 2024 Sorbonne Université, Centre Nationale de la Recherche Scientifique
 
*SpikeFI* is free software: you can redistribute it and/or modify it under the terms of GNU General Public License version 3 as published by the Free Software Foundation.
 
*SpikeFI* is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 
You will found in the LICENSE file a copy of the GNU General Public License version 3.



Since **Slayer** is not available on PyPI and you have it working locally, there are multiple ways to use it in my new project.

---

### **1. Install Slayer as a Local Package**
If your Slayer repo is inside your local machine, navigate to its directory and install it inside your virtual environment --> This is what I did:

```sh
cd /path/to/slayerPytorch
pip install -e .
```

This installs it in **editable mode**, meaning any changes in Slayer’s code will automatically reflect in your project without needing reinstallation.

Then, in your new project, you can simply:

```python
import slayerPytorch as slayer
```

---

### **2. Add Slayer as a Git Submodule**
If you want to keep Slayer inside your new project but still track updates separately:

Inside your new project directory:

```sh
git submodule add https://github.com/<username>/slayerPytorch.git external/slayerPytorch
```

Then install it:

```sh
pip install -e external/slayerPytorch
```

Now, Slayer’s layers will be available to your project.

---

### **3. Directly Clone and Reference Slayer**
If you don’t want to install it but just want to use the modules, copy the `slayerPytorch` directory into your project and import it directly.

For example:
```
my_project/
│── slayerPytorch/  # Copied from the repo
│── main.py
│── requirements.txt
```

Now, in `main.py`:
```python
import slayerPytorch as slayer
```

---

### **4. Use a GitHub Clone in `requirements.txt`**
If your Slayer repo is on GitHub, you can install it directly using:

In `requirements.txt` of your new project:
```
git+https://github.com/<username>/slayerPytorch.git
```

Then run:
```sh
pip install -r requirements.txt
```

---

### **Which Method to Use?**
| Method | Best for |
|--------|---------|
| `pip install -e .` | Local development with easy updates |
| Git submodule | Keeping track of Slayer updates separately |
| Copying repo | Quick and offline usage |
| GitHub install via `requirements.txt` | Using Slayer without manually cloning |


cuda/11.7

python 10.0 >

devtools/11

Torch version:  2.0.1+cu117

CUDA version:  11.7

