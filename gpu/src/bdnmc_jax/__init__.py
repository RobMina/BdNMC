"""Batched (JAX) port of BdNMC.

Every module mirrors one part of stock BdNMC (the reference is RobMina/BdNMC bf45d75; the physics is that of
pgdeniverville/BdNMC master 6cba825) and cites the stock function it ports. Physics runs in float64.
"""
import jax

jax.config.update("jax_enable_x64", True)
