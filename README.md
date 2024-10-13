Based on a fork of the amazing mltrainer and mads_datasets by R. Grouls, 
tytorch is an opinionated base for working with data for and training with pytorch.

The Dataset and DataPipeline follows a strategy pattern that provides for a relatively flat class hierarchy while at the same time allowing for maximum code reuse

The Trainer assumes mlflow for tracking and ray for tuning and is compatible with all the pytorch native dataloaders and objects, as well as the mads_datasets factories and datasets.