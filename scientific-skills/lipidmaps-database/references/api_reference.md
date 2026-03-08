# LIPID MAPS API Reference

This document summarizes common programmatic queries used with the LIPID MAPS database.

Base endpoint:

`https://www.lipidmaps.org/rest`

## Query by LIPID MAPS ID

`/compound/lm_id/LMGP01010001/all`

Typical fields returned include:

- name
- formula
- exact mass
- classification
- structure identifiers

## Query by Abbreviation

`/compound/abbrev/PC(16:0/18:1)/all`

Useful for standardized shorthand-based lookups.

## Query by Exact Mass

`/compound/mass/760.585/all`

Useful for lipidomics annotation and candidate retrieval.

## Structure Fields

Returned structure-related fields may include:

- SMILES
- InChI
- InChIKey

## Bulk Downloads

LIPID MAPS provides downloadable datasets including:

- LMSD structure database
- lipid classifications
- curated lipid lists

## External Cross-References

Cross-references may include:

- HMDB
- KEGG
- PubChem
- ChEBI
