---
name: lipidmaps-database
description: Query, interpret, and standardize lipid information using the LIPID MAPS Structure Database (LMSD) and related lipidomics resources. Supports lipid lookup, classification, structure retrieval, identifier conversion, and lipid nomenclature normalization.
---

# LIPID MAPS Database Skill

## Overview

The **LIPID MAPS Structure Database (LMSD)** is a primary resource for lipidomics.
It provides curated lipid structures, standardized lipid classification, molecular identifiers, and links to pathways and related proteins.

This skill enables AI agents to:

- Retrieve lipid metadata and structures
- Normalize lipid names to **LIPID MAPS shorthand notation**
- Convert between lipid database identifiers
- Classify lipids into standard lipid categories
- Integrate lipid information into lipidomics and metabolomics workflows

The LIPID MAPS classification system divides lipids into eight major categories:

1. Fatty Acyls (FA)
2. Glycerolipids (GL)
3. Glycerophospholipids (GP)
4. Sphingolipids (SP)
5. Sterol Lipids (ST)
6. Prenol Lipids (PR)
7. Saccharolipids (SL)
8. Polyketides (PK)

## When to Use This Skill

Use this skill when a task involves:

- Identifying or annotating lipids from LC-MS/MS experiments
- Converting lipid identifiers (HMDB, PubChem, KEGG, LIPID MAPS)
- Determining lipid class or subclass
- Retrieving SMILES, InChIKey, or structural metadata
- Standardizing lipid names for scientific reporting
- Linking lipid species to metabolic pathways

## Core Capabilities

### 1. Lipid Lookup

Retrieve lipid information using:

- LIPID MAPS ID (`LM_ID`)
- lipid name
- abbreviation
- molecular formula
- exact mass
- SMILES or InChIKey

Typical returned information:

- lipid category and class
- molecular formula
- exact mass
- structure (SMILES/InChI)
- synonyms
- cross-database identifiers

### 2. Lipid Classification

Identify hierarchical lipid classification:

`Category → Class → Subclass → Molecular species`

Example:

`Glycerophospholipid → Phosphatidylcholine → PC(16:0/18:1)`

### 3. Identifier Conversion

Convert between lipid identifiers when cross-references are available:

- LIPID MAPS ↔ HMDB
- LIPID MAPS ↔ PubChem
- LIPID MAPS ↔ KEGG
- LIPID MAPS ↔ ChEBI

### 4. Lipid Structure Retrieval

Retrieve structural information:

- SMILES
- InChI
- InChIKey
- molecular formula
- exact mass
- chemical classification

Useful for:

- cheminformatics
- structure visualization
- docking or modeling workflows

### 5. Lipid Nomenclature Normalization

Normalize lipid names using LIPID MAPS shorthand notation.

Examples:

- `PC 34:1` → `PC(34:1)` at species level
- `PE 18:0_20:4` → `PE(18:0/20:4)` when chain composition is known
- `Cer d18:1/16:0` → `Cer(d18:1/16:0)`

Also supports:

- ether lipids (`O-`, `P-`)
- plasmalogens
- oxidized lipids

### 6. Lipidomics Annotation Support

Assist lipidomics workflows by:

- matching lipid m/z values
- linking lipids to lipid classes
- suggesting possible lipid species
- retrieving reference structures

Works well alongside:

- `pyopenms`
- `matchms`
- `metabolomics-workbench`
- `hmdb-database`

## Example Workflows

### Lipid Identification

1. Obtain precursor m/z from LC-MS/MS
2. Search candidate lipids by mass
3. Retrieve candidate lipid structures
4. Determine lipid class
5. Normalize lipid name

### Lipid Annotation Pipeline

`MS/MS spectrum → matchms / pyOpenMS → candidate mass → LIPID MAPS lookup → lipid class + structure → standardized lipid name`

### Cross-Database Integration

`HMDB lipid ID → convert to LIPID MAPS ID → retrieve lipid structure → map to metabolic pathway`

## Best Practices

### Use Standard Lipid Notation

Prefer LIPID MAPS shorthand notation for reporting lipid species.

Examples:

- species level: `PC(34:1)`
- molecular species level: `PC(16:0/18:1)`

### Distinguish Lipid Resolution

Report lipid resolution correctly:

- species level: `PC(34:1)`
- molecular species: `PC(16:0/18:1)`
- structural level: `PC(16:0/18:1(9Z))`

### Verify Isomers

Multiple lipid species can share the same mass.

Confirm using:

- MS/MS fragments
- retention time
- lipid class fragments

### Track Lipid Classes

Grouping lipids by class helps with:

- pathway analysis
- lipidome interpretation
- statistical comparison

## References

- LIPID MAPS Lipidomics Gateway
- LIPID MAPS Structure Database (LMSD)
- LIPID MAPS lipid classification system
- LIPID MAPS nomenclature guidelines
