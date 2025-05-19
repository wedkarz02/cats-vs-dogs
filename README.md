
# cats-vs-dogs

### Overview

Comparison of different machine learning models in cats and dogs classification.

### Requirements

 - Python 3.10+
 - pip

### Download

Download the source code using the ```git clone``` command:

```bash
$ git clone https://github.com/wedkarz02/cats-vs-dogs.git
```

Or use the *Download ZIP* option from the Github repository [page](https://github.com/wedkarz02/cats-vs-dogs.git).

### Quick Setup

Create a virtual environment:

```bash
$ python3 -m venv venv
```
You might need to install the ```venv``` package in order to do so.

Install required packages in the virtual environment from the ```requirements.txt``` file:

```bash
$ venv/bin/pip3 install -r requirements.txt
```

### Usage

This project requires the training data to be in this exact directory tree:

```
├── data
│   ├── test_set
│   │   ├── cats
│   │   └── dogs
│   └── training_set
│       ├── cats
│       └── dogs
```

Start training the models:

```bash
$ venv/bin/python3 main.py
```

Result graphs will be saved to the ```output/``` directory.

### License

Source code in this project is licensed under the MIT License. See the [LICENSE](https://github.com/wedkarz02/movie_hub/blob/main/LICENSE) file for more info. Images used for model training are licensed under the Apache 2.0 license.
 - [MIT License](https://opensource.org/license/mit)
 - [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

Source of the images: [Kaggle Datasets](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
