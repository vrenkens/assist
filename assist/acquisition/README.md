# Language acquisition model

A language acquisition model maps a spoken utterance onto a task
the model is trained with a number of spoken utterances linked to tasks. Once
trained the acquisition model takes a spoken utterance as input in the form of
a sequence of acoustic feature vectors and returns a task to be performed
by the system assist is controlling. To create your own language acquisition
model you can inherit from the general Model class defined in
model.py and overwrite the abstract methods. Afterwards you should add
it to the factory method in model_factory.py.
