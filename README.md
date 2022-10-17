<p align="center">
<a href="https://github.com/showyourwork/showyourwork">
<img width = "450" src="https://raw.githubusercontent.com/showyourwork/.github/main/images/showyourwork.png" alt="showyourwork"/>
</a>
<br>
<br>
<a href="https://github.com/bruce-edelman/CoveringYourBasis/actions/workflows/build.yml">
<img src="https://github.com/bruce-edelman/CoveringYourBasis/actions/workflows/build.yml/badge.svg?branch=main" alt="Article status"/>
</a>
<a href="https://github.com/bruce-edelman/CoveringYourBasis/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/bruce-edelman/CoveringYourBasis/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
</p>

# Cover Your Basis: Comprehensive Data-Driven Characterization of the Binary Black Hole Population

## [Download Paper PDF](https://github.com/bruce-edelman/CoveringYourBasis/raw/main-pdf/ms.pdf) -- [Download Paper Source](https://github.com/bruce-edelman/CoveringYourBasis/raw/main-pdf/arxiv.tar.gz)

## Instructions to Build Paper

This is an open source scientific article created using the [showyourwork](https://github.com/showyourwork/showyourwork) workflow.
To re-generate the figures macros and paper PDF, clone this repository and run `showyourwork build`

```bash
git clone https://github.com/bruce-edelman/CoveringYourBasis.git
cd CoveringYourBasis
showyouwork build
```

The first build will take a long time as it needs to download some static dataset files from zenodo, but this will be cached after the first build.

## Data

The data is all either hosted on zenodo and will be downloaded or programmatically created in the build process, stored in `src/data/` folder.

## Scripts

Scripts that run the intermediate pipeline steps and read/plot the data to create all figures and macros used in the paper are stored in `src/scripts/`