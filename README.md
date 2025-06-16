# Testing framework for bempp-rsrs

This is a testing framework that interfaces Structured Operators from Python and performs RSRS on them through bempp-rsrs

## Installation

This crate retrieves bempp-cl and KiFMM operators, so both of them should be installed. Here we show an installation through an uv Virtual Environment, but any other python enviroment platform should be suitable for testing. KiFMM 
uses Python 10 and here we repeat the instructions provided in that crate.

1. Begin by installing rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Clone kifmm from https://github.com/bempp/kifmm

2. Install Maturin and pip in a new virtual environment.

```bash
uv venv --python=3.10 && source .venv/bin/activate && uv pip install maturin pip
```

2. Use the Maturin CLI to install the Python bindings into this virtual environment, which additionally will install all required dependencies.

Note that Maturin must be run from the `kifmm` crate root, not the workspace root, and that setuptools must be set to a version compatible with Mayavi.

```bash
pip install 'setuptools<69' .
```

NOTE: If the Mayavi installation fails, it can be removed by commenting the line "mayavi==4.8.1" from the pyproject.toml dependencies in the python folder. This will not affect KiFMM as it is only meant for plotting.

3. Install the Python extension of KiFMM:

```bash
maturin develop --release
```

4. After the installation of kifmm you can install bempp-cl. It is not necessary to install everything from source, except from exafmm:

```bash
git clone https://github.com/exafmm/exafmm-t.git
cd exafmm-t
CFLAGS="-O3" CXXFLAGS="-O3" ./configure
make
make install
python setup.py install
```

To install bempp-cl, in general 

```bash
uv pip install bempp-cl
```

should be enough, but sometimes you will need to install gmsh separately. In this case you should run:

```bash
uv pip install gmsh
```

NOTE: Exafmm will not work o Apple Silicon. To run these operators on your mac, you should remove `assembler="fmm"` when building an operator.



After this installation, you should be good to go and you can run tests from test_rsrs.ipynb. The first cell builds the test case and then you can either run 

```bash
./run_test.sh
```

or run the second cell in the notebook.