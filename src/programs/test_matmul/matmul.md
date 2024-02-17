## forward

$$a^{l+1}_j = \sum{w_{i,j} * a^{l}_i}$$

## backward

$$\dfrac{\partial loss}{\partial a^{l}_i} = \sum{w_{i,j} * \dfrac{\partial loss}{\partial a^{l+1}_j}}$$

$$\dfrac{\partial loss}{\partial w_{i,j}} = {a^{l}_i * \dfrac{\partial loss}{\partial a^{l+1}_j}}$$