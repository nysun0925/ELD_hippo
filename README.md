We noticed that that the ELD framework defines three models in model.py: UniModal, MultiModal, and Model3D.
While we believe our hippocampal (hippo) data are more in line with the definition of multimodal data, we noticed that your provided tutorial exclusively demonstrates UniModal and 3D implementations.
Notably, your multimodal.ipynb example also trains a 3D model instead a multimodal one.

Therefore, we are unsure what kind of parameters should be used to train a multimodal model, and our attempts to train a multimodal model failed to produce results and yielded the following error:

UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
Traceback (most recent call last):
  File "/share/home/yinjun/anaconda3/envs/eld/bin/eld-train", line 8, in <module>
    sys.exit(main())
  File "/share/home/yinjun/anaconda3/envs/eld/lib/python3.8/site-packages/ELD/train.py", line 175, in main
    loss = train_epoch(videoloader, model, myoptimizers, epoch, bSize, args.size)
  File "/share/home/yinjun/anaconda3/envs/eld/lib/python3.8/site-packages/ELD/train.py", line 221, in train_epoch
    output = model.forward(l_iteration)
  File "/share/home/yinjun/anaconda3/envs/eld/lib/python3.8/site-packages/ELD/model.py", line 452, in forward
    inter_mod_loss = 0.5 * ((1 - self.mssi(X_mod1, X_mod1_)) + (1 - self.mssi(X_mod2_, X_mod2)))
  File "/share/home/yinjun/anaconda3/envs/eld/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/share/home/yinjun/anaconda3/envs/eld/lib/python3.8/site-packages/piqa/ssim.py", line 316, in forward
    assert_type(
  File "/share/home/yinjun/anaconda3/envs/eld/lib/python3.8/site-packages/piqa/utils/__init__.py", line 97, in assert_type
    assert value_range[0] <= t.min(), f"Expected all values to be greater or equal to {value_range[0]}, but got {t.min().item()}."
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.



Due to these challenges, the current version of our Jupyter notebook only includes implementations for:
1.UniModal-based landmark detection and rigid registration
2.3D-model-based landmark detection and rigid registration
