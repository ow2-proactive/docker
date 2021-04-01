# docker build -t activeeon/r-ml:gpu .
# docker run -it --rm --gpus all --privileged activeeon/r-ml:gpu bash
# docker run -it --rm --gpus all --privileged activeeon/r-ml:gpu nvidia-smi

# user:pass rstudio:admin
# docker run -it --rm --gpus all --privileged -e PASSWORD=admin -p 8787:8787 activeeon/r-ml:gpu

# For more info:
# https://www.rocker-project.org/
# https://hub.docker.com/r/rocker/ml
# https://github.com/rocker-org/ml

FROM rocker/ml-gpu

# Update linux
RUN apt-get clean && apt-get update -y -qq
RUN apt install -y liblzma-dev libbz2-dev

# Update Java
RUN apt install -y openjdk-8-jdk

# Update R
RUN install2.r -s --error \
    rJava \
    random

# Install caret
RUN install2.r --error \
    minqa \
    nloptr \ 
    RcppEigen \
    profileModel \
    pbkrtest \
    lme4 \
    brglm \
    gtools \
    car \
    BradleyTerry2 \
    caret

# Install related packages
RUN install2.r \
    caretEnsemble \
    ada \
    adabag \
    adaptDA \
    arm \
    bartMachine \
    binda \
    bnclassify \
    brnn \
    bst \
    C50 \
    caTools \
    CHAID \
    class \
    Cubist \
    deepboost \
    deepnet \
    dplyr \
    e1071 \
    earth \
    elasticnet \
    elmNN \
    evtree \
    extraTrees \
    fastAdaboost \
    fastICA \
    FCNN4R \
    foba \
    foreach \
    frbs \
    gam \
    gbm \
    glmnet \
    gpls \
    h2o \
    hda \
    HDclassif \
    HiDimDA \
    import \
    inTrees \
    ipred \
    keras \
    kerndwd \
    kernlab \
    kknn \
    klaR \
    kohonen \
    KRLS \
    lars \
    leaps \
    LiblineaR \
    logicFS \
    LogicReg \
    MASS \
    Matrix \
    mboost \
    mda \
    mgcv \
    monmlp \
    monomvn \
    msaenet \
    mxnet \
    naivebayes \
    neuralnet \
    nnet \
    nnls \
    nodeHarvest \
    obliqueRF \
    ordinalNet \
    pamr \
    partDSA \
    party \
    penalized \
    penalizedLDA \
    pls \
    plsRglm \
    plyr \
    protoclass \
    proxy \
    qrnn \
    quantregForest \
    randomForest \
    randomGLM \
    ranger \
    Rborist \
    relaxo \
    rFerns \
    robustDA \
    rocc \
    rotationForest \
    rpart \
    rpartScore \
    rqPen \
    rrcov \
    rrcovHD \
    RRF \
    rrlda \
    RSNNS \
    RWeka \
    sda \
    sdwd \
    snn \
    sparsediscrim \
    sparseLDA \
    spikeslab \
    spls \
    stepPlr \
    superpc \
    supervisedPRIM \
    vbmp \
    VGAM \
    wsrf \
    xgboost

# Update Python
RUN pip install py4j

# Fix R
# https://github.com/rstudio/rstudio/issues/2254#issuecomment-372474049
RUN R CMD javareconf

# Update Env
ENV R_HOME=/usr/local/lib/R
ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV PATH=$PATH:$HOME/bin:$JAVA_HOME/bin:$R_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$R_HOME/lib:$JAVA_HOME/jre/lib/amd64/server
ENV R_LIBS=$R_HOME/lib

# Clean
RUN apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
