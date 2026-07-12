"""Immutable forecast storage.

Each forecast run should write:

- paths.npz: generated price paths
- metadata.json: model version, data cutoff, random seed, path shape
- features.json: volatility, vol-of-vol, slope, momentum, kurtosis snapshot
"""
