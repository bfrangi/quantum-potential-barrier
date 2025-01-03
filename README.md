# Simulating the Collision of an Electron with a Finite Quantum Potential Square Barrier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14576743.svg)](https://doi.org/10.5281/zenodo.14576743)

Code for *Simulating the Collision of an Electron with a Finite Quantum Potential Square Barrier* developed for a *Quantum Mechanics* course of the UC3M *Engineering Physics* degree.

## Contents

This repository contains the code elaborated mainly in December of 2021 for a quantum physics laboratory. The main file in this repo is ```simulation.py```, which holds the code to solve all the questions posed in ```Documents/Lab Guide.pdf```. It uses some of the functions defined in the file ```quantumpython.py``` and it uses the initial parameters defined in ```initial_parameters.py```.

Additionally, the file ```simulation_euler.py``` is a file that solves the first problem of the Lab Guide using Euler's numerical method. It is just a means to see very easily the way that this method behaves as opposed to the Crank-Nicholson method used in the main file, ```simulation.py```.

The program ```simulation.py``` can generate some ```.txt``` files depending on the options chosen during execution. These have not been included in the repo because of their massive size (they can be generated by following the instructios in the Lab Report, which can be found in in ```Documents/Lab Report.pdf```). However, the other output files (videos and figures) *have* been included. Those can be found in ```Output-Media```.

## Clone this repository
First, make sure you have installed ```git``` on your system (info on how to install [here](https://github.com/git-guides/install-git)). Then, run ```git clone https://github.com/bfrangi/quantum-potential-barrier.git``` to clone this repository into your current working directory.

## Requirements

To run this code on your system you will need:

- Python version: made with python 3.8.5

- Manual installation of these python packages: ```matplotlib```, ```numpy```, ```rich``` and ```progressbar```

## Run

Run the simulation scripts using:

```
python3 <script-name>.py
```

## Cite this repository

```bibtex
@software{bernat_frangi_2024_14576743,
  author       = {Bernat Frangi},
  title        = {Simulating the Collision of an Electron with a Finite Quantum Potential Square Barrier},
  month        = 12,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14576743},
  url          = {https://doi.org/10.5281/zenodo.14576743},
}
```

