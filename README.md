# FMM4RBGPU

[![DOI](https://zenodo.org/badge/535562721.svg)](https://zenodo.org/badge/latestdoi/535562721)

by Yi-Kai Kan (<tw.ykkan@gmail.com>)

This package provides serial and GPU-parallelized fast mutiple method (FMM) routines for the efficient computation of the space-charge field from the relativistic charged particle beam. The FMM is based on the Lagrangian interpolation and dual-tree traversal. Array-based tree data structure is used to implement the cluster tree.

## Installation
The package can be installed using Julia's REPL
```julia
julia> import Pkg
julia> Pkg.add(url="https://github.com/ykkan/FMM4RBGPU.jl.git")
```
or with Pkg mode (hitting `]` in the command prompt)
```julia
pkg> add https://github.com/ykkan/FMM4RBGPU.jl.git
```

## Usage
### Creating a Charged Particle Beam
A charged particle beam can be created by providing an array of particle position and an array of particle momentum.
``` julia
using FMM4RBGPU

# number of particles
const N = 1000    

# create position and momentum distribution of N particles
positions = rand(3, N)
momenta = zeros(3, N)

# create a particle beam in which each particle's charge and mass are -1 and 1 
beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0) 
```

### Space-Charge Field Calculation
The space-charge field of a particle beam can be evaluated using `update_particles_field!` with different algorithms:
* `BruteForce`: the brute-forece method (serial version)
* `FMM`: the proposed FMM (serial version)
* `FMMGPU`: the proposed FMM (GPU-parallelized version)
``` julia
using FMM4RBGPU
# update particle field by FMM with the following parameters: 
#   n: degree of interpolation
#   N0: maximum number of particles in the leaf cluster
#   eta: admissibility parameter 
#   lambda: a characteristic length for the normalization of length quantity
####
const n = 4 
const N0 = 125  
const eta = 0.5
update_particles_field!(beam, BruteForce(); lambda=1.0) 
update_particles_field!(beam, FMM(eta=eta, N0=N0, n=n); lambda=1.0)
update_particles_field!(beam, FMMGPU(eta=eta, N0=N0, n=n); lambda=1.0)
```

After the update of space-charge field of `beam`, the space-charge field acting on the i-th particle can be accessed by
```julia
efield = beam.efields[i]
bfield = beam.bfields[i]
```

## Performance
The elapsed times for the evaluation of space-cahrge field from $2.56\times 10^7$ particles on different hardwares
* CPU Intel Xeon Gold 5218: 3780s (serial)
* GPU Nvidia Telsla P100: 39s (x96 speedup)
* GPU Nvidia Telsla A100: 29.1s (x130 speedup)

## References
* Y.-K. Kan, F. X. Kärtner, S. Le Borne, and J.-P. M. Zemke, A GPU-Parallelized Interpolation-Based Fast Multipole Method for the Relativistic Space-Charge Field Calculation, _Comput. Phys. Commun._ __291__ (2023), 108825.

* Y.-K. Kan, F. X. Kärtner, S. Le Borne, J.-P. M. Zemke, Relativistic Space-Charge Field Calculation by Interpolation-Based Treecode, _Comput. Phys. Commun._ __286__ (2023), 108668.
