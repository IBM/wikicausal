# WikiCausal: Corpus and Task for Evaluation of Causal Knowledge Graph Construction

This repository contains evaluation scripts as well as evaluation results for causal knowledge extraction using the WikiCausal dataset.

## Document Corpus

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7897996.svg)](https://doi.org/10.5281/zenodo.7897996)

The dataset is available on Zenodo: https://doi.org/10.5281/zenodo.7897996

The format is `jsonl`, with each line being a JSON object containing the page contents, meta-data, and associated event concept(s). The fields in each JSON object are:
- `id`: Wikipedia page identifier
- `title`: Wikipedia page title.
- `url`: Wikipedia page URL.
- `document_concept`: The Wikidata concept (instance) associated with the document. It comes with the `QID` and all the labels for the concept, which can be used for causal relation extraction.
- `text`: This is the field that contains the full clean text contents of the Wikipedia article, to be used for causal knowledge extraction.
- `first_section`: A separate field for the first section of the article, as it often contains a summary with all the key causal knowledge. Less scalable methods can use only this field for extraction.
- `categories`: List of categories for the page. Categories can also be useful in identifying the topics covered in a page, which can be useful for the extraction process.
- `infobox`: structured infobox fields and values.
- `headings`: section headings of the Wikipedia page.
- `event_concepts`: The set of top-level event concepts (classes) associated with the page. These are seed event concepts that are superclasses of the `document_concept`.
- `timelines`: Some Wikipedia pages have a timeline section describing sequences of sub-events that occurred during the described event. While not the focus of the evaluation in this paper, such sequences can be mined for causal knowledge, e.g. using event sequence models capable of handling noisy ordered event sequences, such as [Summary Markov Models](https://www.ijcai.org/proceedings/2022/0670.pdf).

## Evaluation Framework

Our evaluation framework consists of:
- A script for recall evalution used a "Base KG" derived from Wikidata: [scripts/recall.py](scripts/recall.py)
- A script for precision evaluation using Large Language Models (LLMs): [scripts/precision.py](scripts/precision.py)
- Data required for evaluation, including Base KG and sample extracted KGs: [data/](data/)
- Results on the latest version of the corpus and Base KG
  - Recall results version 1: [results/recall-v1.md](results/recall-v1.md)
  - Precision results version 1: [results/precision-v1.md](results/precision-v1.md)

## Installation

Requirements: you need Python 3.8+ to run the evaluation scripts. To run with CPU only (tested on a Mac):
```
git clone git@github.com:IBM/wikicausal.git
cd wikicausal
pip install -r requirements.txt
```
To run with GPUs, do `pip install -r requirements-gpu.txt` instead (tested on x86_64 Linux).

## Instructions

To run recall evaluation:
```
python scripts/recall.py
```
To run precision evaluation:
```
python scripts/precision.py
```
Run with `--help` to see a list of arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  -b BASE_KG_FILE, --base_kg_file BASE_KG_FILE
                        Base KG file.
  -i INPUT_KG_FILE, --input_kg_file INPUT_KG_FILE
                        Input automatically-constructed KG for evaluation.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output CSV file.
  -m OUTPUT_MD, --output_md OUTPUT_MD
                        Output Markdown file.
```

## Citation

```
@misc{wikicausal,
      title={WikiCausal: Corpus and Evaluation Framework for Causal Knowledge Graph Construction}, 
      author={Oktie Hassanzadeh},
      year={2024},
      url={https://arxiv.org/abs/2409.00331}, 
}
```

## License

This code base is licensed under the Apache License, Version 2. Separate third-party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1](https://developercertificate.org/) and the [Apache License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache License FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
