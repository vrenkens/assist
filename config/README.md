# Recipes

A recipe contains all the configurations for a certain model trained and tested
on a certain database. Every experiment has several components each with their
own configuration file. A recipe contains the folowing configuration files:

- database.cfg: This file contains all the paths where data should be
read and written. This is the only configuration file that is not included in
the repository because it depends on your setup. You can find more info about
the database configuration in the next section.
- features.cfg: This file contains the configuration for the acoustic features
(e.g. mfcc, fbank, ...). You can  find more information about feature computers
[here](../assist/features).
- coder.cfg: this file contains the configuration about how the semantic
representation is coded into a vector. You can find more information about
coders [here](../assist/tasks).
- acquisition.cfg: This file contains the configuration for the language
acquisitionm model (e.g. NMF). You can find more information about language
acquisition models [here](../assist/acquisition).
- cross_validation.cfg: This file contains the configuration for the
cross_validation. For example how many blocks you want to split your data in.
- train.cgf: configuration for training, e.g. the sections in the database.cfg
you want to use for training.
- test.cgf: configuration for testing, e.g. the sections in the database.cfg
you want to use for testing.
- structure.xml: the semantic structure file. You can find more information
about this file [here](../assist/tasks).

Default configuration files can be created for the coder, the acquisition model
and the feautures. You can find more information about deafault configuration
files [here](## Default configuration)

## Database configuration

The database.cfg file contains one section per speaker with the folowing
options:

- audio: location of the audio file containing a list of all wav files.
Every line in the file contains the name of the utterance and a pointer
to the wav file.
- features (writable): a directory where the features should be stored
- tasks: location of the task file containing a list the tasks
you can find more information on task representations
[here](../assist/tasks). Every line in the file contains the name of the
utterance and the string representing the corresponding task.

your database.cfg should look something like this:

```
[speaker1]
audio = /path/to/speaker1/wav.scp
features = /path/to/speaker1/feature/directory
tasks = /path/to/speaker1/tasks/file

[speaker2]
audio = /path/to/speaker2/wav.scp
features = /path/to/speaker2/feature/directory
tasks = /path/to/speaker2/tasks/file

...
```

## Default configuration

Default configuration files can be created for the coder, the acquisition model
and the feautures. These default configuration files can be found in the
defaults folder in the folder where theses components are defined.

A default configuration contains default values for the configuration files,
if they are not defined in the configuration file, they will be taken from the
default file. The name of the configuration file is the lower case class name
of the class it is meant for.
