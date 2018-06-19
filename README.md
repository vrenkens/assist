# Assist

A project discription of the ASSIST project can be found
[here](https://www.esat.kuleuven.be/psi/projects/current/ASSIST). Assist is
a speech interface that learns to map spoken commands to a semantic
representation. The code is written in Python and most models are built
using TensorFlow.

## Using Assist

Assist works in several stages: data prepation, optional acoustic model training
and training and testing the acquisition model.
Each of these stages uses a recipe for a specific model and database.
The recipe contains configuration files for the all components and defines all
the necesary parameters for the database and the model. You can find more
information on the components in a recipe [here](config).
All of the commands should be run from the repository root directory.

### Data Download

Most of the datasets used to train and test the models can be downloaded from
[here](https://www.esat.kuleuven.be/psi/spraak/downloads/).

### Data preperation

In the data preperation stage all the data is prepared (feature computation).
All the prepared data is written in the directory specified in database.cfg.
Before running the data preperation you should create a database.cfg file in
the recipe directory. You can find more information about the database
configuration [here](config).

You can then do data preperation with:

```
run dataprep <expdir> <recipe> [-c <computing>]
```

with the following arguments:

- expdir: the dicrectory where all created files are written (if any)
- recipe: the path to the recipe directory (may be relative or absolute)
- computing (optional): the  kind of computation used (default: local)

more information about the computing argument can be found [here](computing.md).

### Cross Validation

You can use cross-validation for every speaker seperately.
All the data from a speaker is split into a number of blocks that
have maximally similar semantic content. This is done by minimising the
Jensen-Channon divergene between the blocks.

The model is then trained and tested with an increasing amount of training data.
Starting with one block moving up to all but one blocks of training data.
For each training set size a random number of blocks is selected to use in
the training set. All remaining blocks are used as the test set. To get a more
reliable result this is repeated a number of times for each training set size.
This is called a sub-experiment. For each sub-experiment a new random selection
of blocks is used.

In each sub-experiment the acoustic model is trained or loaded depending on the
configuration, the acquisition model is trained and finally the model is tested.

The models and the result are stored in the expdir. The results are given as
a precision, recal and f1-score.

You can run an experiment with:

```
run cross_validation <expdir> <recipe> [-c <computing>]
```

With arguments as descibed in the Data preparation section.

#### Visualizing results

The results from each sub-experiment are stored in the expdir. However the total
number of sub-experiments for each experiment is quite large. You can plot the
results for a particular experiments with:

```
python plots/plot_database <expdir> [-s 1] [-t 1]
```

With the following arguments:

- expdir: the expdir used for the experiment
- -s (optional): if you want to plot all speaker seperately set to 1 (default 0)
- -t (optional): if you want to plot all results set to 1 if 0 (default) only f1
  will be plotted

If you want to compare the results of multiple experiments you can do:

```
python plots/compare_results.py <result> <expdirs>
```

with the following arguments:

- result: the result you want to plot one of (f1, precision, recal)
- expdirs: a space seperated list of expdirs used in the experiments

### Training

Instead of doing cross-validation you can also just train a model on a trainig
set. The sections in the database.cfg used for training are defined in
train.cfg.

You can train the model with

```
run train <expdir> <recipe> [-c <computing>]
```

### Testing

After training you can test the performance of the model. The sections in the
database.cfg used for tasting are defined in test.cfg.

You can test the model with

```
run test <expdir> <recipe> [-c <computing>]
```

## Designing in assist

Everything in the recipe can be modified (more information about recipes
[here](config)). Most classes used in Assist have a general
class that defines an interface and common functionality for all children and
a factory that is used to create the necessary class. Look into the respective
README files to see how to implement a new class.

In general, if you want to add your own type of class (like a new model) you
should follow these steps:

- Create a file in the class directory
- Create your child class, let it inherit from the general class and overwrite
the abstract methods
- Add your class to the factory method. In the factory method you give your
class a name, this does not have to be the name of the class. You will use
this name in the configuration file for your model so Assist knows which class
to use.
- create a configuration file for your class and put it in whichever recipe you
want to use it for or create your own recipe using your new class.
