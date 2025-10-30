### Experiments Working in Progress (works or don't)
#### 1. Inpainting-Anything / LAMA : NOT WORK
Maybe because of serious occlusion or the strange silhouette of the occlusion mask, inpainting methods like Lama and Inpainting-Anything fail at generating plausible predictions (as shown below).
Instead we use Qwen-Image / Flux-Kontext, the more powerful and general generation framework.

![inpaint_failure](./assets//inpaint_failure.png)
#### 2. Differentiable Rendering
Working in Progress, may refer to this [thread](https://github.com/NVlabs/nvdiffrast/issues/117) for reference.

![nvdrexp](./assets/nvdrexp.jpg)