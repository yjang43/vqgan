# vqgan
Implementation of VQGAN

### debugging VQVAE
vq_loss was always so high.
Needed to check gradient computation graph.
Gradient flow from reconstruction loss should NOT pass on to embedding.
Otherwise, it will greatly change embedding unwanted way.
Even the paper states "The first term is the reconstruction loss (or the data term) which optimizes the decoder and the encoder (through the estimator explained above)".
In otherwords, they do not pass gradient of the first term to embedding.


```
[reconstruction gradient flow should be]
e --x-->z_q --> f(z_q)
       |
z -----+
```

```
[it used to be...]
e ----->z_q --> f(z_q)
       |
z -----|
```

Apparently... more critical issue was duplicated grad for vq_loss instead of rec_loss...

```
[vq loss gradient flow should be]
e ----->z_q --> f(z_q)
       |
z --x--+
```

```
[it used to be...]
e ----->z_q --> f(z_q)
       |
z -----|
```
