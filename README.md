# AIMS-BirdCLEF 🐦

## How to Download Data

Antes de tudo, tem que ter a token kaggle: Kaggle.com -> Settings -> API -> Create Token -> Download 

Além disso, para baixar versões anteriores você tem que aceitar as regras da competição: Kaggle.com -> Competitions -> BirdCLEFXXXX -> Late Submission -> Accept the Rules

Depois, rode o script para baixar o dataset desejado da seguinte maneira.

```
 sh input/birdclef-XXXX/download_birdclefXXXX /path/to/the/FOLDER/in/which/kaggle.json/is/located/
```

O script já colocará tudo nos lugares certo e removerá o .zip após extrair
