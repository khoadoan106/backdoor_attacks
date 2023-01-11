# Implementations of Backdoor Attacks


## Prerequisites
We need the following:
* conda or miniconda (preferred)
* GPU or CPU

## Setup the environment
Clone the repository. The setup script to initialize and activate the environment is collected in `etc/setup_env`. Simply run the following command:
```
. etc/setup_env
```
## Repository artifacts

* `python`: code folder
* `requirements.txt`: list of python reqs
* `README.md`: this doc, and light documentation of this repos.

## Implementations

### Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class (NeurIPS2022).
* [Paper](https://openreview.net/pdf?id=i-k6J4VkCDq)

```diff
Please refer to marksman_conditional_trigger_generation.py and marksman_conditional_backdoor_injection.py
```

### LIRA: Learnable, Imperceptible and Robust Backdoor Attacks (ICCV2021).  
* [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.html)
* Stage 1: Trigger Generation - LIRA learns to generate the trigger in Stage 1. Examples:
    * MNIST
        ```
        . etc/setup_env
       nohup python python/lira_trigger_generation.py --dataset mnist --clsmodel mnist_cnn --path experiments/ --epochs 10  --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit 2>&1 >experiments/logs/mnist_trigger_generation.log &
        ```
    * CIFAR10
        ```
        . etc/setup_env
        nohup python python/lira_trigger_generation.py --dataset cifar10 --clsmodel vgg11 --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit 2>&1 >experiments/logs/cifar10_trigger_generation.log &	
        ```
* Stage 2: Backdoor Injection. After the trigger is learned, LIRA poison and fine-tune the classifier in Stage 2. Examples:
    * MNIST
        ```
        . etc/setup_env
        nohup python python/lira_backdoor_injection.py --dataset mnist --clsmodel mnist_cnn --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit --test_eps 0.01 --test_alpha 0.5 --test_epochs 50 --test_lr 0.01 --schedulerC_lambda 0.1 --schedulerC_milestones 10,20,30,40 2>&1 >experiments/logs/mnist_backdoor_injection.log &	
        ```
    * CIFAR10
        ```
        . etc/setup_env
        nohup python python/lira_backdoor_injection.py --dataset cifar10 --clsmodel vgg11 --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit --test_eps 0.01 --test_alpha 0.5 --test_epochs 500 --test_lr 0.01 --schedulerC_lambda 0.1 --schedulerC_milestones 100,200,300,400 2>&1 >experiments/logs/cifar10_backdoor_injection.log &		
        ```
        
Please cite the paper, as below, when using this repository:
```
@inproceedings{doan2021lira,
  title={Lira: Learnable, imperceptible and robust backdoor attacks},
  author={Doan, Khoa and Lao, Yingjie and Zhao, Weijie and Li, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11966--11976},
  year={2021}
}
```

