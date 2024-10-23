# Lorenzaccio
Character Network of Alfred de Musset's play Lorenzaccio.

# Python Dependencies
You can install dependencies with `pip install -r requirements.txt`. This repository is compatible with Python 3.8, 3.9 and 3.10. Higher versions should work, but are not (yet) explicitely supported by Renard.

# Organization
The `lorenzaccio.tei.xml` file contains the *Lorenzaccio* play under TEI format. It originally comes from project DraCor, with minor manual corrections applied.

The provided Python scripts have the following roles:

| Script             | Role                                                                                                             | Executable |
|--------------------|------------------------------------------------------------------------------------------------------------------|------------|
| `extract.py`       | Extract character networks from the TEI file                                                                     | yes        |
| `comsplot.py`      | Various functions related to plotting and communities                                                            | no         |
| `eng.py`           | Utilities for the English translation of character names                                                         | no         |
| `kcliques_coms.py` | Plot communities obtained with k-cliques percolation, and the associated zp space and normalized prevalence plot | yes        |
| `pplot3.py`        | Pretty plot of the three static character networks (co-occurrence, mention, conversation)                        | yes        |

For any script, you can supply the `-h` option to get information about arguments.
