build_env="dev"
build_date=$(date '+%Y-%m-%d')
build_version="123"
build_tag=${build_env}"-"${build_date}"."${build_version}
echo $build_tag
docker build -t activeeon/sal:$build_tag -f ./Dockerfile --no-cache .