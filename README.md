# AIMS-BirdCLEF 🐦

1. [RECOD Repo Location](#repolocation)
2. [How to Work on RECOD](#work)
3. [How to Pull, Push, or Clone on RECOD](#pull)
4. [How to Download Data of All BirdCLEFs](#download)

## Repo Location on RECOD <a name="repolocation"></a>
Clone your repos on:
```
cd /work/RECOD.USER/AIMS-BirdCLEF
```

## How to Work on RECOD <a name="work"></a>
First build the image (already done on DL-08) (Assumes CUDA 11.0. If different, change base image, and torch versions to match.)
```
sh build_image.sh
```
Then, enter a container on GPU G, port PPPP, and with container name birdclef_container_N by running:
```
sh birdclef_container.sh -g G -p 'PPPP:PPPP' -n N
```
Work!

## How to pull, push or clone this repo on RECOD <a name="pull"></a>
Copy the private repo key to your user folder and change its permissions.
```
cp /work/leonardo.boulitreau/birdclef /home/RECOD.USER/.ssh/     #  substituing RECOD.USER with you recod user
chmod 600 /home/RECOD.USER/.ssh/birdclef  #  substituing RECOD.USER with you recod user
```
Create a configuration File:
```
nano /home/RECOD.USER/.ssh/config #  substituing RECOD.USER with you recod user
```
Ctrl C on the following code, and paste on the editor
```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/birdclef
```
Ctrl X then y then Enter to leave the editor and save.

Either pull 
```
cd /work/RECOD.USER/AIMS-BirdCLEF
git pull
```
or clone wherever you want
```
cd /work/RECOD.USER/
git clone git@github.com:leonardoboulitreau/AIMS-BirdCLEF.git    # Everytime it demands, on pull or push, the password is: tucano
git pull
```
or push
```
git add .
git commit -m 'commit message'
git push
```

## How to Download Data <a name="download"></a>

Antes de tudo, tem que ter a token kaggle.json: Kaggle.com -> Settings -> API -> Create Token -> Download 

Além disso, para baixar versões anteriores você tem que aceitar as regras da competição: Kaggle.com -> Competitions -> BirdCLEFXXXX -> Late Submission -> Accept the Rules

Entre no container, seção [How to Work](#work). Depois, execute:

### Se estiver no RECOD
```
 cd /workspace/aimsbirdclef
 sh input/birdclef-2024/download_birdclef2024.sh /workspace/kaggle-token-leo/
```

### Se estiver no seu PC
```
 sh input/birdclef-XXXX/download_birdclefXXXX /path/to/the/FOLDER/in/which/kaggle.json/is/located/
```

O script já colocará tudo nos lugares corretos e removerá o .zip após extrair.
