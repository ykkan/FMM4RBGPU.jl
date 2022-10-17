# FMM4RBGPU

[![Build Status](https://github.com/ykkan/FMM4RBGPU.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ykkan/FMM4RBGPU.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ykkan/FMM4RBGPU.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ykkan/FMM4RBGPU.jl)


by Yi-Kai Kan (<yikai.kan@desy.de>)

This package provides serial and GPU-parallelized fast mutiple method (FMM) routines for the efficient computation of the space-charge field from the relativistic charged particle beam. The FMM is based on the Lagrangian interpolation and dual-tree traversal. Array-based tree data structure is used to implement the cluster tree. The contruction of this package follows test-driven development (TDD).

__Note__: This repository is not publicly accessible at this moment.

## Speed
The elapsed times for the evaluation of space-cahrge field from $2.56\times 10^7$ particles on different hardwares
* CPU Intel Xeon Gold 5218: 3780s (serial)
* GPU Nvidia Telsla P100: 39s (x96 speedup)
* GPU Nvidia Telsla A100: 29.1s (x130 speedup)

## References
* Y.-K. Kan, F. X. Kärtner, S. Le Borne, and J.-P. M. Zemke, Relativistic Space-Charge Field Calculation by Interpolation-Based Treecode, submitted
   [https://doi.org/10.48550/arXiv.2206.02833](https://doi.org/10.48550/arXiv.2206.02833)

* L. Wilson, N. Vaughn, and R. Krasny, A GPU-accelerated fast 
            multipole method based on barycentric Lagrange interpolation 
            and dual tree traversal, 
	    _Comput. Phys. Commun._ __265__ (2021), 108017. 

* L. Wang, R. Krasny, and S. Tlupova, A kernel-independent treecode 
            based on barycentric Lagrange interpolation, 
	    _Commun. Comput. Phys._ __28__ (2020), 1415-1436.