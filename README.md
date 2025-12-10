# MangaJaNaiTRT

## Overview

MangaJaNaiTRT is a simple command line tool to batch upscale images with ONNX models using TensorRT. It was created as a proof of concept for fast and VRAM-efficient upscaling. If it works well enough its backend will be ported into [MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui). 

## Usage

Python 3.13 is recommended. Once it's installed, clone the repo, set up dependencies and download ONNX models:

<details>
<summary>Windows</summary>

```bash
# Clone the repo
git clone https://github.com/the-database/MangaJaNaiTRT.git

# Navigate to the repo
cd MangaJaNaiTRT

# Create a new Python virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install dependencies (will consume ~3 GB for TensorRT dependencies)
pip install .

# Optional: Download IllustrationJaNai ONNX models
python download_onnx.py
```

</details>

<details>
<summary>Ubuntu / Linux</summary>

```bash
# Clone the repo
git clone https://github.com/the-database/MangaJaNaiTRT.git

# Navigate to the repo
cd MangaJaNaiTRT

# Create a new Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (will consume ~3 GB for TensorRT dependencies)
pip install .

# Optional: Download IllustrationJaNai ONNX models
python3 download_onnx.py
```

</details>

Open `config.ini` and set up the paths to the input image(s), ONNX model, and output path. Then run 

```bash
python main.py
```



First run will build the TensorRT engine which may take several minutes. The `.engine` will be saved to the same directory as the `.onnx` and will be reused on subsequent runs, so the engine does not need to be rebuilt unless different engine settings are required.

To use a different config file, specify it with the `-c` argument:

```bash
python -c otherconfig.ini
```

## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models.

## Related Projects
- [MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui): Windows GUI for manga upscaling with MangaJaNai models, but does not support TensorRT (yet)
- [MangaJaNai](https://github.com/the-database/mangajanai): Main repository for manga upscaling models.
- [traiNNer-redux](https://github.com/the-database/traiNNer-redux): Software for training upscaling models. Includes script for converting PyTorch models to ONNX.
