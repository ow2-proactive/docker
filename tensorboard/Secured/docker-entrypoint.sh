#!/bin/sh
# vim:sw=4:ts=4:et

set -e

if [ -z "$PORT" ]; then
    echo WARNING: The default port 6009 is taken.
else
    sed -i /etc/nginx/sites-available/nginx.conf -e "s/listen [0-9]*;/listen $PORT;/" /etc/nginx/sites-available/nginx.conf
fi

if [ -z "$LOGIN" ] || [ -z "$PASSWORD" ]; then
  echo WARNING: The default login and password are set.
else
  htpasswd -cb /etc/nginx/.htpasswd $LOGIN $PASSWORD
fi

nginx -c /etc/nginx/sites-available/nginx.conf

tensorboard --logdir /logs