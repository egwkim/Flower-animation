# Flower animation

## Purpose

Generate a mathematical animation that traces a flower image by computing a cost-aware path along visually salient boundaries.

## Overview

The project has two primary phases:
- Path computation: detect the flower region, compute a boundary likelihood costmap, and derive a closed path that follows salient edges.
- Animation: convert the path into a mathematical animation.

## Pipeline

1. Image acquisition: supply a single image containing a flower (contrasting color preferred).
2. Segmentation: find the flower region (HSV-based color mask and optional morphology).
3. Edge detection and costmap generation: perform Canny edge detection and produce a static cost per pixel based on the magnitude value.
4. Merging edges: Merge separated edges using multi-source dijkstra and meet-in-the-middle.
4. Path construction: Order the pixels to make a continuous, closed path.
5. Animation: Use FFT to calculate fourier coefficients, and create animation with Manim.

## Notes and design considerations

This project is inspired by 3B1B.

## Credits

Flower image: https://www.pexels.com/photo/photo-of-a-carnation-flower-in-a-glass-vase-15806326/
