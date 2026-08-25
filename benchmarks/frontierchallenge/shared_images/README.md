# FrontierChallenge redistributable CPU image

This directory builds one shared `linux/amd64` image:

- `frontierchallenge/cpu-open:2026.08`: the released redistributable scientific stack,
  distributed only as a compressed Docker archive in the solve Hugging Face dataset,
  used by 81 of the 97 tasks. It contains the general Python/image stack,
  R/Bioconductor/Seurat, AmberTools 24.8, GROMACS, OpenMM, PLUMED, LAMMPS,
  Packmol, CP2K, Phonopy, PySCF, xTB, VMD, Multiwfn, Fiji/ImageJ and the pinned
  Martini files.

It deliberately does **not** contain ORCA. FrontierChallenge does not download,
copy, build, publish, or redistribute ORCA binaries or an image containing
them. The 16 tasks that require ORCA point to the local-only image name
`frontierchallenge/orca-user-local:6.0.1`; evaluators must obtain ORCA under
their own licence and create that private runtime themselves. See
[the ORCA runtime tutorial](../docs/providers/orca.md).

Do not push, export, publish, or share a user-created ORCA image. The local tag
is an interface contract, not a FrontierChallenge release artifact.

## Download for evaluation

```bash
./scripts/setup.sh --track open
```

Setup downloads the archive from Hugging Face, verifies its size, SHA-256 and
image ID, and runs `docker image load`. Docker caches the loaded layers. The
human-readable local tag and immutable identity are recorded in the HF image
manifest and `release/images.json`.

## Build and verify the open image (maintainers)

```bash
./shared_images/build.sh open
./shared_images/verify.sh
```

The image is `linux/amd64` because several included scientific tools are
x86-64 binaries. Docker Desktop can emulate it on Apple Silicon, although
compute-heavy jobs are much slower than on a native x86-64 host.

Fiji is copied from a digest-pinned upstream container image. This lets Docker
cache and resume its registry layers instead of repeatedly fetching one large
archive from the ImageJ download server.

The image contains software only. Each task still runs in a fresh container
with its own input mounted under `/app/input` and its output isolated under
`/app/output`.

## Environment selection

The default Python environment is `general`. AmberTools is isolated and can be
invoked with:

```bash
fc-env amber24 python script.py
fc-env amber24 sander -O ...
```

Common Amber executables (`tleap`, `sander`, `cpptraj`, and others) are also on
the default command path.

## Resource boundary

Give Docker at least 32 GiB of memory before attempting the largest
single-cell and QM/MM tasks; the 8 GiB Docker Desktop assigns by default is not
enough. Task-level CPU, memory, storage and timeout limits remain defined in
each `task.toml`.
