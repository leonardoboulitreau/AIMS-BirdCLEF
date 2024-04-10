# AIMS-BirdCLEF 🐦

## How to Use Repo on RECOD
```
cd /work/leonardo.boulitreau/AIMS-BirdCLEF
```

## How to Download Data

Antes de tudo, tem que ter a token kaggle.json: Kaggle.com -> Settings -> API -> Create Token -> Download 

Além disso, para baixar versões anteriores você tem que aceitar as regras da competição: Kaggle.com -> Competitions -> BirdCLEFXXXX -> Late Submission -> Accept the Rules

Depois, rode o script para baixar o dataset desejado da seguinte maneira.

```
 sh input/birdclef-XXXX/download_birdclefXXXX /path/to/the/FOLDER/in/which/kaggle.json/is/located/
```

O script já colocará tudo nos lugares corretos e removerá o .zip após extrair.

## If Needed, this is How to clone, pull or push this repo on RECOD
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
Ctrl X + y + Enter to leave the editor and save.
Continue...
```
cd /the/folder/where/you/wanna/clone/
git clone git@github.com:leonardoboulitreau/AIMS-BirdCLEF.git    # Everytime it demands, on pull or push, the password is: tucano
git pull
```
Do your work
```
git add .
git commit -m 'commit message'
git push
```
