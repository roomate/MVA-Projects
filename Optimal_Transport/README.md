# Smooth and Sparse Optimal Transport

This repository proposes an implementation from scratch of the work realised in [this paper](https://arxiv.org/abs/1710.06276).
It mainly studies different relaxations of the Kantorovich problem in finite dimension, which I may recall is already in itself the 
relaxation of Monge problem. 

The application considered here is colour transfer, a image processing technique to 'transfer' the colour of a picture to another!

Given a source and a reference image:
<img src="img/arbre.PNG" alt="drawing" width="200"/> <img src="img/drapeau.PNG" alt="drawing" width="200"/> 

we can transport the colour of the first to the second!

<img src="img/cluster_n 10_m=50.PNG" alt="drawing" width="200"/> 