# Genre Analysis of Universal Dependencies

This repository accompanies the SyntaxFest 2021 paper **"How Universal is Genre in Universal Dependencies?"** ([Müller-Eberstein, van der Goot and Plank, 2021](https://personads.me/x/syntaxfest-2021-paper)). After downloading the data and installing the required packages, the paper's experiments can be run using `run-experiments.sh` shell script.

## Installation

Python 3.6+ as well as prerequisite packages are quired to run and evaluate all models. Please install them using the provided requirements file (ideally in a virtual environment):

```bash
(venv) $ pip install -r requirements.txt
```

## Data Setup

### Universal Dependencies

Please first download Universal Dependencies version 2.8 from the [official source](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3687). Unpack all treebanks into the `ud28/treebanks` directory and remove the excluded treebanks `UD_Arabic-NYUAD` and `UD_Japanese-BCCWJ`. Please note that some of the remaining treebanks have raw text that needs to be obtained separately due to licensing restrictions. In the end, the `ud28/treebanks` directory should contain the following treebank directories:

```
UD_Afrikaans-AfriBooms
UD_Akkadian-PISANDUB
UD_Akkadian-RIAO
UD_Akuntsu-TuDeT
UD_Albanian-TSA
UD_Amharic-ATT
UD_Ancient_Greek-PROIEL
UD_Ancient_Greek-Perseus
UD_Apurina-UFPA
UD_Arabic-PADT
UD_Arabic-PUD
UD_Armenian-ArmTDP
UD_Assyrian-AS
UD_Bambara-CRB
UD_Basque-BDT
UD_Beja-NSC
UD_Belarusian-HSE
UD_Bhojpuri-BHTB
UD_Breton-KEB
UD_Bulgarian-BTB
UD_Buryat-BDT
UD_Cantonese-HK
UD_Catalan-AnCora
UD_Chinese-CFL
UD_Chinese-GSD
UD_Chinese-GSDSimp
UD_Chinese-HK
UD_Chinese-PUD
UD_Chukchi-HSE
UD_Classical_Chinese-Kyoto
UD_Coptic-Scriptorium
UD_Croatian-SET
UD_Czech-CAC
UD_Czech-CLTT
UD_Czech-FicTree
UD_Czech-PDT
UD_Czech-PUD
UD_Danish-DDT
UD_Dutch-Alpino
UD_Dutch-LassySmall
UD_English-ESL
UD_English-EWT
UD_English-GUM
UD_English-GUMReddit
UD_English-LinES
UD_English-PUD
UD_English-ParTUT
UD_English-Pronouns
UD_Erzya-JR
UD_Estonian-EDT
UD_Estonian-EWT
UD_Faroese-FarPaHC
UD_Faroese-OFT
UD_Finnish-FTB
UD_Finnish-OOD
UD_Finnish-PUD
UD_Finnish-TDT
UD_French-FQB
UD_French-FTB
UD_French-GSD
UD_French-PUD
UD_French-ParTUT
UD_French-Sequoia
UD_French-Spoken
UD_Frisian_Dutch-Fame
UD_Galician-CTG
UD_Galician-TreeGal
UD_German-GSD
UD_German-HDT
UD_German-LIT
UD_German-PUD
UD_Gothic-PROIEL
UD_Greek-GDT
UD_Guajajara-TuDeT
UD_Hebrew-HTB
UD_Hindi-HDTB
UD_Hindi-PUD
UD_Hindi_English-HIENCS
UD_Hungarian-Szeged
UD_Icelandic-IcePaHC
UD_Icelandic-Modern
UD_Icelandic-PUD
UD_Indonesian-CSUI
UD_Indonesian-GSD
UD_Indonesian-PUD
UD_Irish-IDT
UD_Irish-TwittIrish
UD_Italian-ISDT
UD_Italian-PUD
UD_Italian-ParTUT
UD_Italian-PoSTWITA
UD_Italian-TWITTIRO
UD_Italian-VIT
UD_Italian-Valico
UD_Japanese-GSD
UD_Japanese-Modern
UD_Japanese-PUD
UD_Kaapor-TuDeT
UD_Kangri-KDTB
UD_Karelian-KKPP
UD_Kazakh-KTB
UD_Khunsari-AHA
UD_Kiche-IU
UD_Komi_Permyak-UH
UD_Komi_Zyrian-IKDP
UD_Komi_Zyrian-Lattice
UD_Korean-GSD
UD_Korean-Kaist
UD_Korean-PUD
UD_Kurmanji-MG
UD_Latin-ITTB
UD_Latin-LLCT
UD_Latin-PROIEL
UD_Latin-Perseus
UD_Latin-UDante
UD_Latvian-LVTB
UD_Lithuanian-ALKSNIS
UD_Lithuanian-HSE
UD_Livvi-KKPP
UD_Low_Saxon-LSDC
UD_Makurap-TuDeT
UD_Maltese-MUDT
UD_Manx-Cadhan
UD_Marathi-UFAL
UD_Mbya_Guarani-Dooley
UD_Mbya_Guarani-Thomas
UD_Moksha-JR
UD_Munduruku-TuDeT
UD_Naija-NSC
UD_Nayini-AHA
UD_North_Sami-Giella
UD_Norwegian-Bokmaal
UD_Norwegian-Nynorsk
UD_Norwegian-NynorskLIA
UD_Old_Church_Slavonic-PROIEL
UD_Old_East_Slavic-RNC
UD_Old_East_Slavic-TOROT
UD_Old_French-SRCMF
UD_Old_Turkish-Tonqq
UD_Persian-PerDT
UD_Persian-Seraji
UD_Polish-LFG
UD_Polish-PDB
UD_Polish-PUD
UD_Portuguese-Bosque
UD_Portuguese-GSD
UD_Portuguese-PUD
UD_Romanian-ArT
UD_Romanian-Nonstandard
UD_Romanian-RRT
UD_Romanian-SiMoNERo
UD_Russian-GSD
UD_Russian-PUD
UD_Russian-SynTagRus
UD_Russian-Taiga
UD_Sanskrit-UFAL
UD_Sanskrit-Vedic
UD_Scottish_Gaelic-ARCOSG
UD_Serbian-SET
UD_Skolt_Sami-Giellagas
UD_Slovak-SNK
UD_Slovenian-SSJ
UD_Slovenian-SST
UD_Soi-AHA
UD_South_Levantine_Arabic-MADAR
UD_Spanish-AnCora
UD_Spanish-GSD
UD_Spanish-PUD
UD_Swedish-LinES
UD_Swedish-PUD
UD_Swedish-Talbanken
UD_Swedish_Sign_Language-SSLC
UD_Swiss_German-UZH
UD_Tagalog-TRG
UD_Tagalog-Ugnayan
UD_Tamil-MWTT
UD_Tamil-TTB
UD_Telugu-MTG
UD_Thai-PUD
UD_Tupinamba-TuDeT
UD_Turkish-BOUN
UD_Turkish-FrameNet
UD_Turkish-GB
UD_Turkish-IMST
UD_Turkish-Kenet
UD_Turkish-PUD
UD_Turkish-Penn
UD_Turkish-Tourism
UD_Turkish_German-SAGT
UD_Ukrainian-IU
UD_Upper_Sorbian-UFAL
UD_Urdu-UDTB
UD_Uyghur-UDT
UD_Vietnamese-VTB
UD_Warlpiri-UFAL
UD_Welsh-CCG
UD_Western_Armenian-ArmTDP
UD_Wolof-WTB
UD_Yoruba-YTB
UD_Yupik-SLI
```

### Data Splits

The `ud28/splits` directory contains two split definitions Python pickles. They  contain dictionaries of the form `{"train": [0, 1, ...], "dev": [100, 101, ...], "test": [500, 501, ...]}` indicating which sentences in UD (referenced by absolute index) are relevant for the experiments and belong to which split.

* `ud28/splits/102-915-204.pkl` contains the global 204k test split (unaltered test portions of all treebanks), 915k dev split and 102k train split.
* `ud28/splits/71-30-0.pkl` contains the 71k train and 30k dev split which were constructed by further splitting the 102k training split.

### Metadata and Mappings

* `ud28/meta.json` contains metadata for all treebanks as well as parsing instructions for instance metadata.
* `ud28/tb-genres.json` contains mappings from treebank specific genre to global UD genre labels for 26 treebanks.

## Run Experiments

After all relevant packages have been installed and the data has been set up, all experiments can be run by executing the `run-experiments.sh` script. Individual experiments can be re-run by using the appropriately commented lines in the script above.
