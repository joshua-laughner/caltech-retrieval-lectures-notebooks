# Retrieval theory supporting notebooks

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/joshua-laughner/caltech-retrieval-lectures-notebooks/HEAD)

This repo contains notebooks that support the retrieval theory lectures for Caltech's ESE 144 class.
Use the Binder badge above to launch this notebook online.
To run locally:

1. Create a Python virtual environment (`python -m venv /path/to/venv`)
2. Activate that environment (`source /path/to/env/bin/activate`)
3. Install the requirements (`pip install -r requirements.txt` from the root of this repo)
4. Ensure that the `ipykernel` package is also installed (`pip install ipykernel`)
5. Install the environment as a notebook kernel (`ipython kernel install --user --name=retrieval-theory-ese144`)
6. Launch JupyterLab or Jupyter Notebook, open the notebook of interest, and ensure it uses the kernel you just installed

If the widgets in the notebook do not work, see the [ipywidgets documentation](https://github.com/jupyter-widgets/ipywidgets/blob/main/docs/source/user_install.md).
In particular, you may need to ensure that the `jupyterlab_widgets` (for JupyterLab) or `widgetsnbextension` (for Jupyter Notebook) package is installed in the same environment as your notebook server.
