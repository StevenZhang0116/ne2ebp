# Without End-to-end Backpropagation

This work serves for the final project in CSE 543 Deep Learning (UW). [Final Presentation](https://stevenzhang0116.files.wordpress.com/2023/12/cse543_final_presentation.pdf), [Final Report](https://stevenzhang0116.files.wordpress.com/2023/12/cse543_report.pdf). 

This repository summarizes some demo applications (collected online, mostly on MNIST dataset) of learning algorithms that could be considered as alternatives to BP. The credits are reserved to the original authors, in which my contributions are mainly the revisions that guarantee the codes are runnable and comparable.

The included algoritms are: 
* [Feedback Alignment](https://www.nature.com/articles/ncomms13276)
* [Direct Feedback Alignment](https://arxiv.org/abs/1609.01596)
* [Equilibrium Propagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full)
* [(Difference) Target Propagation](https://arxiv.org/abs/1412.7525)
* [HSIC Bottleneck](https://arxiv.org/abs/1908.01580)
* [ADMM](https://arxiv.org/abs/1605.02026)

Use Conda to run code in each demo: 

```
conda create --name test_ne2ebp --file conda_requirements.yml python=3.8
```