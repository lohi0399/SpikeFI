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

> T. Spyrou, S. Hamdioui and H.-G. Stratigopoulos, "SpikeFI: A Fault Injection Framework for Spiking Neural Networks," 2024, preprint ⟨hal-04825966⟩

```bibtex
@unpublished{spyrou:hal-04825966,
  title = {{SpikeFI: A Fault Injection Framework for Spiking Neural Networks}},
  author = {Spyrou, T. and Hamdioui, S. and Stratigopoulos, H.-G.},
  url = {https://hal.science/hal-04825966},
  note = {preprint},
  year = {2024},
  month = Dec,
  keywords = {Neuromorphic Computing ; Neuromorphic Computing ; Spiking Neural Networks ; Reliability ; Fault Simulation ; Testing ; Fault Tolerance},
  pdf = {https://hal.science/hal-04825966v1/file/SpikeFI__A_Fault_Injection_Framework_for_Spiking_Neural_Networks.pdf},
  hal_id = {hal-04825966},
  hal_version = {v1},
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

Copyright 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6

*SpikeFI* is free software: you can redistribute it and/or modify it under the terms of GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

*SpikeFI* is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
