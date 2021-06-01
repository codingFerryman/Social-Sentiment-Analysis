# Errors

## Error in `experiment.py` 1

```
[2021-05-27 17:24:49] - RobertaModel - {line:117} DEBUG - tfOrPyTorch=torchOrTFEnum.TF
[2021-05-27 17:24:49] - RobertaModel - {line:119} DEBUG - training tf model
2021-05-27 17:24:50.584778: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-05-27 17:24:50.584821: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-05-27 17:24:50.584877: I tensorflow/core/profiler/lib/profiler_session.cc:158] Profiler session tear down.
Epoch 1/2
WARNING:tensorflow:The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
WARNING:tensorflow:The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
WARNING:tensorflow:From /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/ops/array_ops.py:5043: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
Instructions for updating:
The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.
WARNING:tensorflow:AutoGraph could not transform <bound method TFRobertaClassificationHead.call of <transformers.models.roberta.modeling_tf_roberta.TFRobertaClassificationHead object at 0x7f32147bd2e0>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: invalid syntax (tmpdvrts1rz.py, line 10)
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
Traceback (most recent call last):
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/src/experimentConfigs/experiment.py", line 123, in <module>
    main(sys.argv)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/src/experimentConfigs/experiment.py", line 119, in main
    launchExperimentFromJson(testPath, reportPath)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/src/experimentConfigs/experiment.py", line 98, in launchExperimentFromJson
    launchExperimentFromDict(experimentSettings, reportPath)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/src/experimentConfigs/experiment.py", line 77, in launchExperimentFromDict
    evals = model.testModel(**d['args'])
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/src/experimentConfigs/../models/transformersModel.py", line 133, in testModel
    self.model.fit(train_dataset.prefetch(2), epochs=num_epochs,
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/training.py", line 1183, in fit
    tmp_logs = self.train_function(iterator)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 872, in __call__
    result = self._call(*args, **kwds)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 916, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 753, in _initialize
    self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/function.py", line 3050, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/function.py", line 3279, in _create_graph_function
    func_graph_module.func_graph_from_py_func(
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 662, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/framework/func_graph.py", line 986, in wrapper
    raise e.ag_error_metadata.to_exception(e)
TypeError: in user code:

    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/training.py:855 train_function  *
        return step_function(self, iterator)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/training.py:845 step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/distribute/distribute_lib.py:1285 run
        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/distribute/distribute_lib.py:2825 call_for_each_replica
        return self._call_for_each_replica(fn, args, kwargs)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/distribute/distribute_lib.py:3600 _call_for_each_replica
        return fn(*args, **kwargs)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/training.py:838 run_step  **
        outputs = model.train_step(data)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/training.py:800 train_step
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/compile_utils.py:439 update_state
        self.build(y_pred, y_true)
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/keras/engine/compile_utils.py:361 build
        self._metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects,
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/util/nest.py:1374 map_structure_up_to
        return map_structure_with_tuple_paths_up_to(
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/util/nest.py:1456 map_structure_with_tuple_paths_up_to
        assert_shallow_structure(
    /home/sniper/projects_local/CIL/Computational-Intelligence-Lab/venv/lib64/python3.9/site-packages/tensorflow/python/util/nest.py:1060 assert_shallow_structure
        raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(

    TypeError: The two structures don't have the same sequence type. Input structure has type <class 'tuple'>, while shallow structure has type <class 'transformers.modeling_tf_outputs.TFSequenceClassifierOutput'>.
```

The problem is that the metric has mismatched input for the dataset and the model. E.g. using `accuracy` rather than `SparseCatergoricalAccuracy` for roberta.