"""Reusable implementations between core contracts and workflow plugins.

The layering invariant is ``core`` ← ``components`` ← ``workflows``;
components must not import workflow code.
"""
