"""Simulation kernels (JAX)."""

# Bumped whenever kernel numerics change (facet ordering / blocking /
# fusion rewrites); chunk-cache keys carry it so stale caches re-simulate.
KERNEL_VERSION = "2026-09-03-global-facets"
