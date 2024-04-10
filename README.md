# AIMS-BirdCLEF 🐦

## How to Use Repo on RECOD
```
cd /work/leonardo.boulitreau/AIMS-BirdCLEF
```
## How to clone, pull or push on RECOD

```
cp /work/leonardo.boulitreau/birdclef /home/RECOD.USER/.ssh/     #  substituing RECOD.USER with you recod user
```

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
Continue...
```
cd /the/folder/where/you/wanna/clone/
git clone git@github.com:leonardoboulitreau/AIMS-BirdCLEF.git    # Everytime it demands, on pull or push, the password is: tucano
git pull
# do your work
git add .
git commit -m 'commit message'
git push
```
## How to Download Data

Antes de tudo, tem que ter a token kaggle.json: Kaggle.com -> Settings -> API -> Create Token -> Download 

Além disso, para baixar versões anteriores você tem que aceitar as regras da competição: Kaggle.com -> Competitions -> BirdCLEFXXXX -> Late Submission -> Accept the Rules

Depois, rode o script para baixar o dataset desejado da seguinte maneira.

```
 sh input/birdclef-XXXX/download_birdclefXXXX /path/to/the/FOLDER/in/which/kaggle.json/is/located/
```

O script já colocará tudo nos lugares certo e removerá o .zip após extrair.
