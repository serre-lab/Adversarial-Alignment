
<div style="text-align:center" class="latex-font">
    <h1 style="text-align: center; font-weight: bold; color: inherit; margin-bottom: 0.2em"> Adversarial Alignment: breaking the trade-off between the strength of an attack and its relevance to human perception </h1>

    <span class="author" style="font-weight: bold"> Drew Linsley*<sup>1, 3</sup>, Pinyuan Feng*<sup>2</sup>, Thibaut Boissin<sup>4</sup>, Alekh Karkada Ashok<sup>1, 3</sup>, Thomas Fel<sup>1, 3, 4</sup>, Stephanie Olaiya<sup>1</sup>, Thomas Serre<sup>1, 2, 3, 4</sup> <br> 
    </span> <br>
    <span class="affiliations"> Department of Cognitive, Linguistic, & Psychological Sciences, Brown University, Providence, RI, USA </span> <br>
    <span class="affiliations"> Department of Computer Science, Brown University, Providence, RI, USA </span> <br>
    <span class="affiliations"> Carney Institute for Brain Science, Brown University, Providence, RI, USA </span> <br>
    <span class="affiliations"> Artificial and Natural Intelligence Toulouse Institute (ANITI), Toulouse, France </span> <br>
    </span> <br>

    <!-- <p align="right"> 
    <i> * : all authors have contributed equally. </i>
    </p> -->
    <!-- <span class="mono"> {drew_linsley, pinyuan_feng}@brown.edu</span> -->
</div>

<p align="center">
  <a href=""><strong>Read our paper »</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://github.com/serre-lab/Adversarial-Alignment"><strong>View our GitHub »</strong></a>
  <br>
  <br>
  <!-- <a href="https://github.com/serre-lab/Adversarial-Alignment">GitHub</a>
  · -->
  <a href="https://serre-lab.github.io/Adversarial-Alignment/results">Results</a>
  ·
  <a href="https://serre-lab.github.io/Adversarial-Alignment/models/">Model Info</a>
  ·
  <a href="https://arxiv.org/abs/2211.04533">Harmonization</a>
  ·
  <a href="https://arxiv.org/abs/1805.08819">ClickMe</a>
  ·
  <a href="https://serre-lab.clps.brown.edu/">Serre Lab @ Brown</a>
</p>

<!-- <div>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
    <a href="https://github.com/serre-lab/Harmonization/actions/workflows/python-lint.yml">
        <img alt="PyLint" src="https://github.com/serre-lab/Harmonization/actions/workflows/python-lint.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Harmonization/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/serre-lab/Harmonization/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Harmonization/actions/workflows/python-pip.yml">
        <img alt="Pypi" src="https://github.com/serre-lab/Harmonization/actions/workflows/python-pip.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/harmonization">
        <img alt="Pepy" src="https://pepy.tech/badge/harmonization">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div> -->


## **Abstract**

<p align="center">
<img src="./assets/teaser.png" width="80%" align="center">
</p>

Deep neural networks (DNNs) are known to have a fundamental sensitivity to adversarial attacks, perturbations of the input that are imperceptible to humans yet powerful enough to change the visual decision of a model. Adversarial attacks have long been considered the "Achilles' heel" of deep learning, which may eventually force a shift in modeling paradigms. Nevertheless, the formidable capabilities of modern large-scale DNNs have somewhat eclipsed these early concerns. Do adversarial attacks continue to pose a threat to DNNs?

In this study, we investigate how the robustness of DNNs to adversarial attacks has evolved as their accuracy on ImageNet has continued to improve. We measure adversarial robustness in two different ways: First, we measure the smallest adversarial attack needed to cause a model to change its object categorization decision. Second, we measure how aligned successful attacks are with the features that humans find diagnostic for object recognition. We find that adversarial attacks are inducing bigger and more easily detectable changes to image pixels as DNNs grow better on ImageNet, but these attacks are also becoming less aligned with the features that humans find diagnostic for object recognition. To better understand the source of this trade-off and if it is a byproduct of DNN architectures or the routines used to train them, we turn to the neural harmonizer, a DNN training routine that encourages models to leverage the same features humans do to solve tasks. Harmonized DNNs achieve the best of both worlds and experience attacks that are both detectable and affect object features that humans find diagnostic for recognition, meaning that attacks on these models are more likely to be rendered ineffective by inducing similar effects on human perception. Our findings suggest that the sensitivity of DNNs to adversarial attacks can be mitigated by DNN scale, data scale, and training routines that align models with biological intelligence. We release our code and data to support this goal.

<!-- ## **Qualitative Results**

<p align="center">
<img src="./assets/qualitative.png" width="100%">
</p>

**$\ell_2$ PGD adversarial attacks for DNNs.** Plotted here are ImageNet images, human feature importance maps from ClickMe, and adversarial attacks for a variety of DNNs. Attacked images are included for the image of a monkey at the top (zoom in to see attack details). The red box shows inanimate categories, and the blue box shows animate categories. -->

<!-- ## Authors

<p align="center">

<div class="authors-container">

  <div class="author-block">
    <img src="./assets/thomas.png" width="25%" align="center">
    <a href="mailto:thomas_fel@brown.edu"> Thomas Fel* </a>
  </div>


  <div class="author-block">
    <img src="./assets/ivan.png" width="25%" align="center">
    <a href="mailto:ivan_felipe_rodriguez@brown.edu"> Ivan Felipe Rodriguez* </a>
  </div>


  <div class="author-block">
    <img src="./assets/drew.png" width="25%" align="center">
    <a href="mailto:drew_linsley@brown.edu"> Drew Linsley* </a>
  </div>

  <div class="author-block">
    <img src="./assets/tserre.png" width="25%" align="center">
    <a href="mailto:thomas_serre@brown.edu"> Thomas Serre </a>
  </div>

</div>

<br>
<p align="right"> 
<i> * : all authors have contributed equally. </i>
</p>

</p> -->


## **Citation**

If you use or build on our work as part of your workflow in a scientific publication, please consider citing the [official paper]():

```
@article{linsley2023adv,
  title={Adversarial Alignment: breaking the trade-off between the strength of an attack and its relevance to human perception},
  author={Linsley, Drew and Feng, Pinyuan and Boissin, Thibaut and Ashok, Alekh Karkada and Fel, Thomas and Olaiya Stephanie and Serre, Thomas},
  year={2023}
}
```

If you have any questions about the paper, please contact Drew at [drew_linsley@brown.edu](drew_linsley@brown.edu).

## **Acknowledgement**

This paper relies heavily on previous work from [Serre Lab](https://serre-lab.clps.brown.edu/), notably [Harmonization](https://serre-lab.github.io/Harmonization/) and [ClickMe](https://serre-lab.clps.brown.edu/resource/clickme/).

```
@article{fel2022aligning,
  title={Harmonizing the object recognition strategies of deep neural networks with humans},
  author={Fel, Thomas and Felipe, Ivan and Linsley, Drew and Serre, Thomas},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@article{linsley2018learning,
  title={Learning what and where to attend},
  author={Linsley, Drew and Shiebler, Dan and Eberhardt, Sven and Serre, Thomas},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```

<!-- ## Tutorials

**Evaluate your own model (pytorch and tensorflow)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mp0vxUcIsX1QY-_Byo1LU2IRVcqu7gUl) 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/230px-Tensorflow_logo.svg.png" width=35>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bttp-hVnV_agJGhwdRRW6yUBbf-eImRN) 
<img src="https://pytorch.org/assets/images/pytorch-logo.png" width=35> -->


## **License**

The code is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.