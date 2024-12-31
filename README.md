# Rendez le script exécutable avec la commande suivante

 chmod +x build_docker.sh

 # Maintenant, vous pouvez exécuter ce script à chaque fois que vous voulez reconstruire l'image Docker 

 ./build_docker.sh

 # Une fois que le script a terminé son exécution, vérifiez que l'image a bien été créée en exécutant la commande suivante 

 docker images

 # Rendez le script exécutable

 chmod +x run_docker.sh

 # Maintenant, lancez le container en exécutant le script 

 ./run_docker.sh

 # Ouvrez votre navigateur web et allez à l'adresse http://localhost:8000. Vous devriez voir l'interface utilisateur simple de notre application LLM

 http://localhost:8000



Excellent ! Vous avez maintenant votre application LLM en cours d'exécution dans un container Docker. Voici comment utiliser l'application :

Préparation des données d'entraînement :

Créez un fichier texte contenant vos données d'entraînement. Chaque ligne devrait contenir une phrase ou un court paragraphe.
Sauvegardez ce fichier sur votre ordinateur.


Utilisation de l'interface web :
a) Uploader et prétraiter les données :

Cliquez sur le bouton "Choisir un fichier" sous "Upload and Preprocess".
Sélectionnez votre fichier de données d'entraînement.
Cliquez sur "Upload and Preprocess".
Attendez le message de confirmation.

b) Entraîner le modèle :

Cliquez sur le bouton "Train Model".
Attendez le message de confirmation. Cela peut prendre un certain temps selon la taille de vos données.

c) Générer du texte :

Dans le champ de texte sous "Generate Text", entrez une phrase de départ.
Cliquez sur "Generate Text".
Le texte généré apparaîtra en dessous.


Quelques points à noter :

Le modèle est très simple et l'entraînement est rapide, donc les résultats peuvent être de qualité variable.
Vous pouvez répéter le processus d'upload et d'entraînement avec différents ensembles de données.
Si vous fermez le container Docker, toutes les données et le modèle entraîné seront perdus. Vous devrez recommencer le processus lors du prochain démarrage.