# sapiensph

A simple [PCISPH](https://doi.org/10.1145/1576246.1531346) implementation using [NVIDIA Warp](https://github.com/NVIDIA/warp) and [SAPIEN](https://github.com/haosulab/SAPIEN). This is my homework project for the course [CSE 291 (SP 2024 C00) Physics Simulation](https://cseweb.ucsd.edu/~alchern/teaching/cse291_sp24/) by Prof. Albert Chern at UCSD. This implementation is very simple and was done in less than a day, including reading the paper and coding. 

## Installation

### Prerequisites

- Python 3.8 or later
- CUDA 12 or later

### Step 1: Install SAPIEN 3

Use pip to install the latest SAPIEN 3 wheel from [SAPIEN Nightly Release](https://github.com/haosulab/SAPIEN/releases/tag/nightly). Look for your own python version. 

The installation command should look like this:

```sh
pip install https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev{SOME_DATE}-cp{PYTHON_VERSION}-cp3{PYTHON_VERSION}-manylinux2014_x86_64.whl
```

See the [SAPIEN](https://github.com/haosulab/SAPIEN) repo for reference.

### Step 2: Install sapiensph

Clone this repo and install it using pip:

```sh
git clone git@github.com:Rabbit-Hu/sapiensph.git
cd sapiensph
pip install .
```

## Usage

See the `examples/example.py` for a simple example. 

```sh
python examples/example.py
```

## References

- B. Solenthaler and R. Pajarola. 2009. Predictive-corrective incompressible SPH. In ACM SIGGRAPH 2009 papers (SIGGRAPH '09). Association for Computing Machinery, New York, NY, USA, Article 40, 1–6. https://doi.org/10.1145/1576246.1531346

