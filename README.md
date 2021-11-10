# Emotion Recognition met CNN

Deze folder bevat de realisatie voor de webapp demonstrator van emotion recognition met een Convolutional Neural Network (CNN). Deze app is beschikbaar gesteld in een docker container format voor vereenvoudigde deployment en gebruik.
Ook bevat deze folder het python programma waarmee het CNN is getraind. Deze kan worden gebruikt om een eventueel nieuw model te bouwen.

## Docker Setup (Webapp)

De app wordt gelanceerd met twee containers: Een nginx container die het webverkeer regelt, en een flask-socket-io container die het convolutional neural network bevat. Deze is zowel aangeleverd met voorgebouwd image, als een variant waarin het originele flask-socket-io wordt gebouwd vanaf het begin af aan. Dit als eventuele backup voor als een van de twee opties niet functioneren.

### Pre-Build

Open de folder "cnn-webapp-docker (pre-build)" in een command prompt.
Laadt vervolgens de voorgebouwde images in met het commando "docker load -i compose-images.tar"
Dit proces duurt even voordat deze tekst toont dat het laden geslaagd is.
Voer vervolgens het commando "docker-compose up" of "docker compose up" uit (afhankelijk van welke versie van Docker is geïnstalleerd is).

### From-Scratch

Open de folder "cnn-webapp-docker (pre-build)" in een command prompt.
Voer vervolgens het commando "docker-compose up" of "docker compose up" uit (afhankelijk van welke versie van Docker is geïnstalleerd is).

## CNN Training

De CNN trainer maakt een nieuwe CNN model aan met keras tensorflow. Deze maakt gebruikt van DirectML waarmee als een grafische kaart beschikbaar is, deze ook wordt gebruikt om het model te helpen trainen. Het programma is **niet** dynamisch opgesteld, waarmee wordt bedoeld dat het netwerk een vaste topology heeft. Het programma is echter wel instaat om andere datasets te accepteren en te oversampelen of undersampelen.

NOTE: De trainer is nooit bedoeld geweest voor productie doeleinden.

### Gebruik: Virtual Environment

Voor de CNN trainer is Python 3.6 gebruikt. Als deze is geïnstalleerd, open vervolgens de "tensor-directml" folder in een command prompt.
Voer vervolgens met de juiste python versie (3.6) het commando "python -m venv .venv" voor Windows of "python3 -m venv .venv" voor Linux/Unix.
Activeer de virtuele omgeving die is aangemaakt en voer vervolgens het commando "pip install -r requirements.txt" uit in deze virtuele omgeving.
Als eenmaal deze omgeving is aangemaakt kun je de omgeving gebruiken om main.py te runnen zoals je een normaal python programma zou runnen.

NOTE: Als je het programma andere modellen wil laten train zal het programma moeten worden aangepast, omdat deze statisch is.