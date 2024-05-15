build_env="dev"
build_date=$(date '+%Y-%m-%d')
build_tag=${build_env}"-"${build_date}
echo $build_tag
docker build -t activeeon/sal:$build_tag -f ./Dockerfile --no-cache .