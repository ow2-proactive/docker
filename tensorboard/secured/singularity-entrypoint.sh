#!/bin/sh
# vim:sw=4:ts=4:et

LOCAL_PATH="$4"

sed -i "$LOCAL_PATH"/.nginx/nginx_singularity.conf -e "s@auth_basic_user_file .nginx@auth_basic_user_file $LOCAL_PATH/.nginx@" "$LOCAL_PATH"/.nginx/nginx_singularity.conf

if [ -z "$1" ]; then
    echo WARNING: The default port 6009 is taken.
else
    sed -i "$LOCAL_PATH"/.nginx/nginx_singularity.conf -e "s/listen [0-9]*;/listen $1;/" "$LOCAL_PATH"/.nginx/nginx_singularity.conf
fi

if [ -z "$2" ] || [ -z "$3" ]; then
  echo WARNING: The default login and password are set.
else
  htpasswd -cb "$LOCAL_PATH"/.nginx/.htpasswd "$2" "$3"
fi

#echo nginx -c "$LOCAL_PATH"/.nginx/nginx_singularity.conf
nginx -c "$LOCAL_PATH"/.nginx/nginx_singularity.conf

tensorboard --logdir /logs