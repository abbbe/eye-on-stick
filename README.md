```
ENV=eos
conda create -y -n $ENV python=3.6
conda activate $ENV
pip install -r requirements.txt
```

If you have Visual Studio Code, at this point you can already run eye-on-stick.ipynb, see rendered env view and growing eye-on-stick.log.
Otherwise install and run Jupyter Lab, for example:

Dev (optional)
```
conda install -y -c conda-forge nodejs
pip install -r requirements-dev.txt
jupyter labextension install nbdime-jupyterlab jupyter-matplotlib

jupyter lab
```