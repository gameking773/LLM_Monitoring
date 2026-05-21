# Benchmark des modèles automatisé

Utiliser une lib permettant de benchmarker un modèle avec des datasets publics, par exemple : 

https://huggingface.co/datasets/stanfordnlp/coqa
https://huggingface.co/datasets/openai/openai_humaneval

Ensuite monitorer le noeud de calcul lors de l'execution du dataset. 

Comparer les résultats entre modèles. Objectifs actuel : 

Vérifier si le moteur d'inférence (vllm ou llama.cpp) a une influence sur la qualité des réponses et sur la consommation.  
Vérifier les différencesentre un modèle non quantifié et le même modèle mais quantifié.  
Vérifier si l'encodage en gguf a une influence sur la qualité du modèle et sa consommation.  
Faire tout ça sur plusieurs modèle de plusieurs entreprises différentes.