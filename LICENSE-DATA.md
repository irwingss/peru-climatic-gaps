# Licence for figures, tables and derived data

The figures (`figures/`), tables (`tables/`), supplementary files (`supplementary/`) and derived data (`data/`) of this repository are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, with the exception below.

## Files derived from IUCN Red List and BirdLife data

The IUCN Red List and BirdLife International terms allow non-commercial use of works derived from their range maps and require attribution. The following files derive from those ranges and are licensed under CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/, with attribution to the IUCN Red List of Threatened Species (version 2026-1) and to BirdLife International and Handbook of the Birds of the World (2025):

- `supplementary/File_S1_species_IUCN_BirdLife.xlsx`
- `tables/supplementary/TableS10.csv`
- `tables/supplementary/TableS11.csv`, `TableS12.csv`, `TableS13.csv` and `tables/main/Fig4_prioritization_table.csv` (species scores)
- `figures/main/Fig4_species_vs_climate.png` and `figures/main/Graphical_abstract.png`
- `figures/supplementary/FigS2_workflow.png` (thumbnail of the species ranges), `FigS12_species_shares_continental.png` and `FigS13_species_shares_maritime_*.png`
- `data/prioritization/` (species features, planning-unit amounts of each species, solutions and their species scores)

The range maps themselves, and the File S3 maps drawn from them, are not part of this repository. File S3 is distributed with the article.

## Third-party inputs

The inputs used to produce these files keep their own terms and are not redistributed here: CHELSA, Bio-ORACLE, PISCOp, SERNANP, MINAM, INEI, MapBiomas Peru, GHSL, HydroBASINS, WorldClim, the IUCN Red List, BirdLife International and SENAMHI. `data/insumos_sha256.json` lists the exact files used and `clima_local/descargar_insumos.py` downloads those that can be downloaded. The country outlines in `figuras_tablas/insumos_figuras/paises.geojson` come from Natural Earth (public domain).
