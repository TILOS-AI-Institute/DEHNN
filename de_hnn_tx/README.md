## Instructions for setup

# 1. Data Location
Please put all the new design directories to "de_hnn_tx/data/cross_design_data/"

# 2. Data Process
Go to directory "de_hnn_tx/data/" and Run the following command 
```commandline
python gen_data.py eigen split part
```

This will automatically create corresponding new features for each design. 

# 3. Data Load and Test Model
We prepare an example to load new data and test our trained model at [test example notebook](test_example.ipynb)

# 4. Train the new model
If you would like to train the model based on some new data, we also prepare an example at [train example notebook](test_example.ipynb)


# More about models
We also prepared all the available models in "de_gnn_tx/models/layers/"

Notice that, "de_gnn_tx/models/model.py" is a general framework we used to load different models, but not the exact implementation for each layer. 

Thank you!
