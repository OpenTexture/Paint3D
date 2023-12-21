

<div align="center">
    <h1> <a>Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models</a></h1>

<p align="center">
  <a >Project Page</a> •
  <a >Arxiv</a> •
  Demo •
  <a href="#️-faq">FAQ</a> •
  <a href="#-citation">Citation</a>
</p>

</div>


https://github.com/OpenTexture/Paint3D/assets/18525299/9aef7eeb-a783-482c-87d5-78055da3bfc0


##  Introduction

Paint3D is a novel coarse-to-fine generative framework that is capable of producing high-resolution, lighting-less, and diverse 2K UV texture maps for untextured 3D meshes conditioned on text or image inputs。

<details open="open">
    <summary><b>Technical details</b></summary>

We present Paint3D, a novel coarse-to-fine generative framework that is capable of producing high-resolution, lighting-less, and diverse 2K UV texture maps for untextured 3D meshes conditioned on text or image inputs. The key challenge addressed is generating high-quality textures without embedded illumination information, which allows the textures to be re-lighted or re-edited within modern graphics pipelines. To achieve this, our method first leverages a pre-trained depth-aware 2D diffusion model to generate view-conditional images and perform multi-view texture fusion, producing an initial coarse texture map. However, as 2D models cannot fully represent 3D shapes and disable lighting effects, the coarse texture map exhibits incomplete areas and illumination artifacts. To resolve this, we train separate UV Inpainting and UVHD diffusion models specialized for the shape-aware refinement of incomplete areas and the removal of illumination artifacts. Through this coarse-to-fine process, Paint3D can produce high-quality 2K UV textures that maintain semantic consistency while being lighting-less, significantly advancing the state-of-the-art in texturing 3D objects.

<img width="1194" alt="pipeline" src="./assets/images/pipeline.jpg">
</details>

## 🚩 News

- [2023/12/21] Upload paper and init project 🔥🔥🔥

## ⚡ Quick Start

<!-- <details>
  <summary><b>Setup and download</b></summary>

</details> -->

## ▶️ Demo

<!-- <details>
  <summary><b>Webui</b></summary>


</details> -->

## 👀 Visualization

## ⚠️ FAQ

<details> <summary><b>Question-and-Answer</b></summary>
    

</details>
</details>

## 📖 Citation



## Acknowledgments

Thanks to [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper), 
[Text2Tex](https://github.com/daveredrum/Text2Tex), 
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet), our code is partially borrowing from them. 
Our approach is inspired by [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [Michelangelo](https://neuralcarver.github.io/michelangelo/) and [DreamFusion](https://dreamfusion3d.github.io/).

## License

This code is distributed under an [Apache 2.0 LICENSE](LICENSE).

Note that our code depends on other libraries, including [PyTorch3D](https://pytorch3d.org/) and [PyTorch Lightning](https://lightning.ai/), and uses datasets which each have their own respective licenses that must also be followed.
