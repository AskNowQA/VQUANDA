# VQuAnDa - Verbalization Question Answering Dataset

## Introduction
We introduce a KBQA dataset containing verbalizations of answers. The dataset is based on [LC-QuAD](https://github.com/AskNowQA/LC-QuAD) which uses [DBpedia v04.16](https://wiki.dbpedia.org/dbpedia-version-2016-04) as the target KB.

## VQuAnDa details
The dataset contains 5000 examples and we have already split it in train (80%) and test (20%) sets.

* `dataset/` here you will find the dataset files (train, test).

The dataset is stored in JSON dumps and each instance contains 4 key-value pairs:
```bash
{
    "uid": "Unique id in the dataset",
    "question": "Question",
    "verbalized_answer": "Answer verbalization",
    "query": "SPARQL query of the question"
}
```

## Baseline models
Alongside the dataset, we provide some baseline models. [Here](https://github.com/endrikacupaj/VQUANDA-Baseline-Models) you can find the baseline implementations and instructions for how to run them.

## License
The dataset is under [Attribution 4.0 International (CC BY 4.0)](LICENSE)

## Cite
```bash
@InProceedings{kacupaj2020vquanda,
    title={VQuAnDa: Verbalization QUestion ANswering DAtaset},
    author={Kacupaj, Endri and Zafar, Hamid and Lehmann, Jens and Maleshkova, Maria},
    booktitle={The Semantic Web},
    pages={531--547},
    year={2020},
    publisher={Springer International Publishing},
}
```