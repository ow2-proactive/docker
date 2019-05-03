FROM activeeon/dlm3

WORKDIR /root

RUN pip install -U proactive --upgrade
RUN pip install -U proactive-jupyter-kernel --upgrade

RUN rm -rf ~/.cache/pip

# http://jupyter-notebook.readthedocs.io/en/stable/security.html
ADD jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

ADD proactive_python_client.ipynb /root

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
