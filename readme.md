# Radiation Fault Injection on DPU-Based CNN Inference (ZCU104)

This project studies the effect of radiation-induced memory faults (DDR4 bit
upsets and SEFI events) on the accuracy of a ResNet50 image classification
model running on a Xilinx Deep Learning Processing Unit (DPU). Faults are
injected directly into the DDR4 physical memory regions used by the DPU
(weights, instructions, input tensor, output tensor, and intermediate feature
maps) via `/dev/mem`, and the resulting change in top-1 classification
accuracy is measured and logged.

All work in this repository has been done on the hardware/software stack
described below. Migration to a Versal-based board has **not** been done yet
and is out of scope for this README.

## Hardware / Software Platform

- **Board:** Xilinx ZCU104 evaluation board
- **DPU IP:** `DPUCZDX8G` (PL-fabric based DPU, two cores — core addressed
  at base `0x80000000` onward for core 1, per `controlRegisters.cc`)
- **Model:** ResNet50 image classifier
  - Originally the Caffe ResNet50 model from the Xilinx Model Zoo
  - Later re-quantized/compiled from a PyTorch ResNet50 model (see
    `sefi_results_pt/` vs `sefi_results/` below)
- **Software stack:** Vitis-AI (VART runtime, `xir`, `vai_c_xir`), OpenCV,
  running on the ZCU104's PetaLinux target
- **Memory:** DDR4, accessed from the ARM host side via `/dev/mem` at the
  physical addresses exposed through the DPU's AXI control registers

## Repository Structure

```
.
├── application_code/     # Main C++ applications run on the ZCU104 target
├── controlRegisters/     # Tool to dump raw DPU control-register values
├── files/                 # Shared helper code and label/reference files
├── ref/                    # Reference weight/instruction binaries dumped from the xmodel
├── train_subset/           # Subset of ImageNet validation images used for testing
└── Radiation-Faults/       # All experiment results, logs, and plots
```

### `application_code/`

| File | Purpose |
|---|---|
| `main.cc` | Baseline ResNet50 inference application (no fault injection). Used to establish clean accuracy on `train_subset` and confirm the model/preprocessing is correct before running any fault campaign. |
| `ddr4_verify.cc` | Verifies that the physical DDR4 addresses read from the DPU control registers (`dpu_base0_addr` for weights, `dpu_instr_addr` for instructions, etc.) actually correspond to the correct regions in DDR4, by comparing `/dev/mem` readback against the reference binaries in `ref/`. This was the validation step used to confirm the base addresses were correct before trusting them for fault injection. It also documents (in its header comments) the findings that shaped the fault injection code — e.g. that the instruction register returns a DPU-local/IOMMU address rather than a direct CPU physical address, and that the input tensor DDR4 region has a 2080-byte VART header before the actual pixel data. |
| `MBU_simulate.cc` | Multi-Bit Upset (MBU) fault injection: flips a configurable number of bits in a chosen DDR4 region (weights, input tensor, buffers/feature maps) for each test image, runs inference, records accuracy and probability drop, then restores the original bytes. |
| `SEFI_simulate.cc` | Single Event Functional Interrupt (SEFI) fault injection implementing multiple SEFI fault patterns (row, column, block, and their transient variants) as described in the DDR4 SEFI literature, applied to the full (non-split) DPU model. |
| `SEFI_transient.cc` | Transient SEFI fault injection using a version of ResNet50 split into per-block pieces, so that a fault can be injected into one block's weights, that block alone executed, and the fault restored before continuing — enabling true transient (single-inference-window) fault behavior at block granularity. |

### `controlRegisters/`

| File | Purpose |
|---|---|
| `controlRegisters.cc` | Reads and prints the raw DPU AXI control register values (instruction address, weight/feature-map/input/output tensor base addresses, done/status registers, etc.) for both DPU cores during inference. Used to determine the physical DDR4 base addresses for each memory region — the addresses that `ddr4_verify.cc` and the fault-injection tools then use to locate weights, instructions, and tensors in DDR4. |
| `log.txt` | Captured raw output from running `controlRegisters.cc`. |

### `files/`

Shared helper code and reference/label files used across the applications above:

| File | Purpose |
|---|---|
| `common.h` / `common.cpp` | Shared helper functions (from the Vitis-AI ResNet50 example code) used by the applications for pre/post-processing and model I/O. |
| `synset.txt` / `words.txt` | ImageNet class label files, used to map model output indices to human-readable class names. |
| `xclbinutil.txt` | Captured output of `xclbinutil --info --input dpu.xclbin`, used as the source for the DPU register offset map (see the register table below). |

### `ref/`

Reference binaries dumped directly from the compiled `resnet50.xmodel` using
`xir dump_bin`, used as ground truth for `ddr4_verify.cc` and for sizing
the fault-injection targets:

| File | Purpose |
|---|---|
| `REG_0.bin` | Weights binary (~25.7 MB), the reference copy of what should reside at the DPU's weights DDR4 base address. |
| `subgraph_conv1.mc` | Instructions binary (~742 KB), the reference copy of the DPU instruction stream. |
| `readme.txt` | Notes on how these two files were generated (`xir dump_bin` command) and what they contain. |

### `train_subset/`

A subset of ImageNet images (organized by WordNet synset ID folders, e.g.
`n01748264/`, `n02093991/`, ...) used as the test set for all accuracy
measurements in this project.

### `Radiation-Faults/`

All results generated by the fault-injection campaigns, plus the plotting
scripts used to visualize them:

| Folder | Contents |
|---|---|
| `controlRegistersMethod/` | Early MBU results (`MBU_simulate01.cc` plus per-target `buffers/`, `feature_maps/`, `weights/` subfolders), each with `plot_results.py` and generated `plot_*.png` accuracy/probability-drop plots. |
| `mbu_results/` | MBU fault injection results (CSV + plots) for `weights/`, `input_tensor/`, and `buffers/` targets, using the current `MBU_simulate.cc`. |
| `sefi_results/` | SEFI fault injection results on the original **Caffe** ResNet50 model, one subfolder per SEFI mode (`01. SEFI-row`, `02. transient-SEFI-row`, `03. SEFI-column`, `04. transient-SEFI-col`, `05. SEFI-block`, `06. transient-SEFI-blk`), each containing logs and per-target (`weights/`, `input_tensor/`, `buffers/`) results. |
| `sefi_results_pt/` | Same SEFI experiments as above, but run on the **PyTorch**-derived ResNet50 model (confirmed by the tensor names in the logs, e.g. `ResNet__input_0_fix` instead of the Caffe model's `data_fixed`). |
| `sefi_transient_results/` | Results from `SEFI_transient.cc` (the block-split transient SEFI experiments), covering the `transient-SEFI-row` and `transient-SEFI-blk` modes at weight granularity. |
| `sefi_plot.py` | Plotting script for the SEFI result sets. |

## DPU DDR4 Memory Map (reference)

The following register map and sizing information (used throughout the
fault-injection code to locate each memory region in DDR4) was determined
from `xclbinutil --info` and `xir dump_reg` / `xir dump_bin` on the target,
and is documented in detail in the project's control-register notes:

| Region | Register offset(s) | Notes |
|---|---|---|
| Instructions | LO @ `0x50`, HI @ `0x54` (from core base `0x80000000`) | Register returns a page index; multiply by 4096 (left-shift by 12) to get the DDR4 address. Size: 742,492 B (~742 KB). |
| Weights (REG_0 / CONST) | `dpu_base0_addr` @ `0x60`/`0x64` | Direct 1:1 CPU physical address. Size: 25,726,976 B (~25.7 MB). |
| Feature maps (REG_1 / WORKSPACE) | `0x68`–`0x6F` | Size: 2,207,744 B (~2.1 MB). |
| Input tensor (REG_2 / INTERFACE) | `0x70`–`0x77` | Size: 152,608 B. VART prepends a 2080-byte (`0x820`) header before the actual pixel data, so pixel data starts at `base + 2080`. |
| Output tensor (REG_3 / INTERFACE) | `0x78`–`0x7F` | Size: 1,008 B — padded from the 1000 class logits to a 16-byte AXI transfer boundary (1000 → 1008 = 63 × 16). |

Control register buses are 32-bit, while DDR4 addresses are 64-bit; each
64-bit address is therefore split across two adjacent 32-bit registers
(LO/HI, 4 bytes apart) and recombined as `{HI, LO}`.

## References

The following official Xilinx/AMD documentation was used as reference
material for Vitis kernel integration and DPU platform creation:

1. [Vitis-Tutorials — Mixing C and RTL Kernels](https://github.com/Xilinx/Vitis-Tutorials/blob/2023.2/Hardware_Acceleration/Feature_Tutorials/02-mixing-c-rtl-kernels/README.md)
2. [Vitis-AI — DPUCZ Reference Design (Vitis flow)](https://github.com/Xilinx/Vitis-AI/blob/3.0/dpu/ref_design_docs/README_DPUCZ_Vitis.md)

## Repository

[https://github.com/bikramghub12345/Versal-Project](https://github.com/bikramghub12345/Versal-Project)

---

**Note:** This README reflects work done on the ZCU104/DPUCZDX8G platform
only. Migration or porting of this work to a Versal ACAP board is a planned
future step and is not covered here.
