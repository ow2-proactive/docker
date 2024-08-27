#!/bin/bash

# Generate the hashed password
HASHED_PASSWORD=$(python3 -c "from jupyter_server.auth import passwd; print(passwd(passphrase='$JUPYTERLAB_PASSWORD', algorithm='sha1'))")

# Execute JupyterLab with additional options
exec jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --ServerApp.password="$HASHED_PASSWORD" $ADDITIONAL_OPTIONS
