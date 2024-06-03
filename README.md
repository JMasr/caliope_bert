# caliope_bert
The Caliope Project is a comprehensive toolkit developed by the [Multimedia Technologies Group](https://gtm.uvigo.es/en/)  at the **atlanTTic Research Center, Universidade de Vigo**. This project, conducted under the TILGA initiative in collaboration with the [Cluster Audiovisual Galego](https://www.clusteraudiovisualgalego.com/), aims to enhance transcriptions by adding capitalization and punctuation.

The **Caliope-Toolkit** currently supports Spanish and Galician languages, with the flexibility to incorporate additional languages in the future.

This repository provides an implementation of the Caliope Toolkit using a BERT model.

## Repository's environment setup using conda.
$ conda env create --name caliope --file environment.yml

$ conda activate caliope

$ pip install -r requirements.txt

## How to:
From a cmd:
A default call just needs as an arguments:

* ```-i -> path to the directory with the file.eaf and the wordconfid.txt```
* ```--language -> two options gl or es (gl is the default value)```

Example:
```
(caliope):$ python elan2cvs2srt.py -i /path/to/file.eaf --language gl
```
