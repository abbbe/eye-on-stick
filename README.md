Local install:
```
ENV=eos
conda create -y -n $ENV python=3.6
conda activate $ENV
pip install -r requirements.txt
```

At this point you can already run eye-on-stick.ipynb (for instance in Visual Studio Code).
MLFlow will stash some metrics and tensorboard events under mlruns/.

To view MLFlow GUI at http://localhost:5000/ run:
```
mlflow ui --backend-store-uri sqlite:///mlruns/db.sqlite
```

To view Tensorboard at http://localhost:6006/ run:
```
tensorboard --logdir mlruns/
```

To make a decent local development environment, install and run Jupyter Lab locally:
```
conda install -y -c conda-forge nodejs
pip install -r requirements-dev.txt
# (unstable?) jupyter labextension install nbdime-jupyterlab jupyter-matplotlib
jupyter lab
```