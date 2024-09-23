#!/bin/bash
set -e

# Run your initialization tasks here

echo "Running redis server"
redis-server --daemonize yes --protected-mode no
sleep 1
redis-cli ping
sleep 1

# echo "Running triton proxy (js)"
# # (cd /opt/triton_proxy/ && node triton_proxy.js &)
# (cd /opt/triton_proxy/ && nodemon triton_proxy.js &)
# sleep 1

echo "Running triton proxy (py)"
# (cd /opt/triton_proxy/ && python3 triton_proxy.py &)
(cd /opt/triton_proxy/ && ./triton_proxy.py --debug &)
sleep 1

echo "Running drift monitoring service (py)"
# (cd /opt/triton_proxy/ && python3 drift_detection.py &)
(cd /opt/triton_proxy/ && ./drift_detection.py &)
sleep 1

echo "Running fastapi server"
# (cd /opt/api_server/ && uvicorn server:app --host 0.0.0.0 --port 9000 --app-dir /opt/api_server/ --root-path /api/ --forwarded-allow-ips "*" --reload &)
(cd /opt/api_server/ && uvicorn server:app --host 0.0.0.0 --port 9000 --app-dir /opt/api_server/ --forwarded-allow-ips "*" --reload &)
sleep 1

echo "Running code-server"
if [ ! -z "${PASSWORD}" ]; then
    code-server --log error --disable-telemetry --disable-update-check --disable-workspace-trust --auth=password --bind-addr=0.0.0.0:9999 /opt/ &
else
    code-server --log error --disable-telemetry --disable-update-check --disable-workspace-trust --auth=none --bind-addr=0.0.0.0:9999 /opt/ &
fi
sleep 1

echo "Running entry-command"
# Check if a command was passed
if [ $# -eq 0 ]; then
    echo "Running bash"
    # If no command was passed, run your default command
    exec bash
else
    echo "Running $@"
    # If a command was passed, execute it
    exec "$@"
fi
