FROM jupyter/scipy-notebook

USER root

RUN mkdir work/output
COPY input/ work/input
COPY notebooks/ work/notebooks
ADD requirements.txt .

RUN pip install -r requirements.txt && fix-permissions /home/$NB_USER

USER $NB_UID
