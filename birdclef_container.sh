while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container birdclef_container_$number on gpu $gpu and port $port";
docker run --rm -it --gpus device=$gpu --userns=host --shm-size 64G -v /work/leonardo.boulitreau/kaggle-token-leo/:/workspace/kaggle-token-leo/ -v $PWD:/workspace/aimsbirdclef/ -p $port --name birdclef_container_$number aims-birdclef:latest /bin/bash
