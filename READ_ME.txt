Le projet est découpé en 3 fichiers :

Maze_env.py : contient la classe MazeEnv qui permet de créer un environnement de labyrinthe.
    La fonction à changer est "generate_obstacles" : elle reonvoie une liste d'obstacles à placer dans le labyrinthe.
    Format d'une liste de tuples : ex [(1,2), (3,4), (5,6)]

Q_learning.py : contient les fonctions (basées sur l'algorithme QLearning) qui permet d'entrainer un agent à trouver la sortie du labyrinthe.

Main.py : contient le code qui permet de lancer l'entrainement de l'agent.
    La constante Q permet de définir si on veut lancer l'entrainement ou pas (True ou False)
        Utile pour voir comment est affiché le labyrinthe sans l'entrainement (qui risque de planter si le labyrinthe n'est pas adapté : typiquement si l'agent ne peut jamais atteindre le goal)


Pour lancer le projet, il suffit de lancer le fichier Main.py