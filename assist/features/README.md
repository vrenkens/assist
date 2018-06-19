# Feature Computers

a feature computer is used to compute audio features. It takes a signal as input
and returns a sequence of vectors in a matrix. To create your own feature
computer you can inherit from the general FeatureComputer class defined in
feature_computer.py and overwrite the abstract methods.
Afterwards you should add it to the factory method in
feature_computer_factory.py.
