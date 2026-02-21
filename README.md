# Unsupervised Classification of Sentinel-3 Altimetry Echoes

## Contents

- [Project Background](#project-background)
- [Objectives](#objectives)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)


## Project Background 

### Why Collocation Matters

In Earth Observation, many applications link measurements from different sensors. This requires collocation, meaning matching observations in:

- **Space** (overlapping footprint)
- **Time** (same time or within an acceptable window)

Collocation is non-trivial because sensors differ in:

- Spatial resolution  
- Sampling geometry  
- Revisit time  

For example:

- **Sentinel-2 imagery** provides a 2D fixed grid (Eulerian frame) at ~10 m resolution.  
- **Sentinel-3 altimetry** samples along a 1D moving track (Lagrangian frame).  

This results in a mismatch in spatial dimensions when comparing datasets. Echo classification is therefore commonly combined with collocated datasets (such as satellite imagery) to support training, validation, and interpretation.
