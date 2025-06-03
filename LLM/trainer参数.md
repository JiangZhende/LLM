## Trainer
### Parameters
**model** (PreTrainedModel or torch.nn.Module, optional)-用于训练、测试或者预测的模型。如果不提供，**model_init**一定要提供。

**args** (TrainingArguments, Optional)-训练参数。如果为提供，则默认为TrainingArguments的基础实例，**output_dir**默认为当前目录下的*tmp_trainer*。

**data_collator**（DataCollator，optional）-从**train_dataset**或者**eval_dataset**中生成批处理格式的函数。如果没有提供processing_class，则默认为default_data_collator()；如果processing_class是特征提取器或标记器，则默认为DataCollatorWithPadding实例。

**train_dataset**（Union[torch.utils.data.Dataset,torch.utils.data.IterableDatset],optional）- 用于训练的数据集。如果它是一个Dataset，不被model.forward()方法接受的列将被自动删除。

请注意，如果它是带有一些随机化的torch.utils.data. iterabledataset，并且您正在以分布式方式进行训练，则您的可迭代数据集应该使用内部属性generator（即torch.Generator）。用于随机化的生成器必须在所有进程中相同（并且Trainer将在每个epoch手动设置此生成器的种子）或具有set_epoch（）方法，该方法在内部设置所使用的rngs的种子。

**eval_dataset**（Union[torch.utils.data.Dataset,torch.utils.data.IterableDatset],optional）- 用于评估的数据集。如果它是一个Dataset，不被model.forward()方法接受的列将被自动删除。如果它是一个字典，它将在每个数据集进行测试并将字典的值添加到度量名称。

**processing_class** (PretrainedTokenizerBase or BaseImageProcessor or FeatureExtractionMixin or ProcessorMixin, optional)- 用于处理数据集的类。如果提供，则使用它来自动处理数据集用户模型，并且它将保存在模型中，便于继续进行中断的训练或者重新训练微调的模型。这将取代tokenizer参数，这个参数将不再被使用。

**model_init** (Callable[[], PreTrainedModel], optional)- 用于初始化模型的函数。如果提供，则使用它来初始化模型，而不是使用model参数。

**compute_loss_func** (Callable, optional)- 一个接收原始模型输出、标签和整个累加批处理的数据（batch_size*gradient_accumulated_steps）的函数，并且返回loss。

**compute_metrics** (Callable[[EvalPrediction], Dict], optional) - 这个函数用户计算评估指标。

**callbacks** (List of TrainerCallback, optional)- 自定义训练循环的回调列表。

**optimizers** (Tuple[Optimizer, SchedulerType], optional)- 用于训练的优化器和学习率调度器。如果提供，则使用它来初始化优化器和学习率调度器，而不是使用args.optim。

**preprocess_logits_for_metrics** (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)- 该函数用于在每一个求值步骤缓存logits之前对其进行处理的函数。接收原始模型输出和标签的函数，并且返回处理后的输出。这个函数所做的修改将来用于compute_metrics函数。

## TrainingArguments
**output_dir** (str, optional)- 保存模型预测和检查点的目录。

**overwrite_output_dir** (bool, optional)- 如果为True，则覆盖output_dir。如果output_dir保存了checkpoint，则使用此命令进行继续预训练。

**do_train** (bool, optional)- 是否进行训练。这个参数不直接被Trainer使用，它是由你的训练/测试脚本使用。

**do_eval** (bool, optional)- 是否进行评估。如果eval_strategy不是no，则设置为True。这个参数不直接被Trainer使用，它是由你的训练/测试脚本使用。

**do_predict** (bool, optional)- 是否在测试集上执行预测。这个参数不直接被Trainer使用，它是由你的训练/测试脚本使用。

**eval_strategy** (str, optional)- 评估策略。可选值为：no, steps, epochs。

**prediction_loss_only** (bool, optional)- 如果为True，在执行评估和生成预测时，只返回预测的损失。

**per_device_train_batch_size** (int, optional)- 每个设备上的训练批处理大小。

**per_device_eval_batch_size** (int, optional)- 每个设备上的评估批处理大小。

**gradient_accumulation_steps** (int, optional)- 梯度累积步数。

**eval_accumulation_steps** (int, optional)- 评估累积步数。

**eval_delay** (int, optional)- 评估延迟步数。

**torch_empty_cache_steps** (int, optional)- 在训练过程中，每隔多少步释放缓存。

**learning_rate** (float, optional)- 学习率。

**weight_decay** (float, optional)- 权重衰减。

**adam_beta1** (float, optional)- Adam优化器的beta1参数。

**adam_beta2** (float, optional)- Adam优化器的beta2参数。

**adam_epsilon** (float, optional)- Adam优化器的epsilon参数。

**max_grad_norm** (float, optional)- 梯度裁剪的最大范数。

**num_train_epochs** (float, optional)- 训练的epoch数。

**max_steps** (int, optional)- 训练的最大步数。覆盖num_train_epochs。

**lr_scheduler_type** (str, optional)- 学习率调度器类型。可选值为：linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup。

**lr_scheduler_kwargs** (Dict, optional)- 学习率调度器的额外参数。

**warmup_ratio** (float, optional)- 预热步数占总步数的比例。

**warmup_steps** (int, optional)- 预热步数。覆盖warmup_ration。

**log_level** (str, optional) - 要在主进程上使用的日志级别。可能的选择是作为字符串的日志级别：‘ debug ’， ‘ info ’， ‘ warning ’， ‘ error ’和‘ critical ’，加上一个‘ passive ’级别，它不设置任何内容，并保持当前的日志级别为transformer库（默认为‘ warning ’）。

**log_level_replica** (str, optional)- 在副本上使用的日志记录，级别与log_level相同。

**log_on_each_node** (bool, optional)- 在分布式训练时，是否在每个节点记录日志，还是在主节点记录。

**logging_dir** (str, optional)- TensorBoard记录目录。

**logging_strategy** (str, optional)- 训练期间使用的日志策略。no：不记录；epoch：在epoch结束后记录；steps：每个logging_steps记录。

**logging_first_step** (bool, optional)- 是否记录第一个global_step。

**logging_steps** (int or float, optional)- 如果logging_strategy=steps，则在两个日志之间更新步骤。值在0到1，如果小于1，将被解释为训练步数的比例。

**logging_nan_inf_filter** (bool, optional)- 是否过滤nan和inf在记录的时候。

**save_strategy** (str or SaveStrategy, optional)- checkpoint的保存策略。no：不保存；epoch：每个epoch结束保存；steps：每个save_steps保存；best：每当达到best_metric时保存。

**save_steps** (int or float, optional)- 如果save_strategy=steps，两个checkpoint之间的步数。

**save_total_limit** (int, optional)- 限制保存的checkpoint。

**save_sagetensors** (bool, optional)- 使用safetensors保存或加载状态字典，而不是torch。

**save_on_each_node** (bool, optional)- 在分布式训练时，是否在每个node保存模型。

**save_only_model** (bool, optional)- 是否只保存模型，或者还有优化器，调度器，rng状态。

**restore_callback_states_from_checkpoint** (bool, optional)- 是否从检查点回复回调状态。

**use_cpu** (bool, optional)- 是否使用cpu。

**seed** (int, optional)- 在开始训练前设置的随机种子。

**data_seed** (int, optional)- 用于数据取样器的随机种子。

**jit_mode_eval** (bool, optional)- 是否使用PyTorch jit跟踪推理。

**use_ipex** (bool, optional)- 是否使用Intel扩展。

**bf16** (bool, optional)- 是否使用bf16。

**fp16** (bool, optional)- 是否使用fp16.

**fp16_opt_level** (str, optional)- 使用fp16，Apex AMP优化等级从[O0， O， O2，O3]中选择。

**fp16_backend** (str, optional)- 这个参数被删除。

