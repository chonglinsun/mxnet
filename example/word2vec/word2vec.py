import os, sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np

def is_param_name(name):
    return name.endswith('weight')\
              or name.endswith('bias')\
              or name.endswith('gamma')\
              or name.endswith('beta')

class SkipGram(object):
    def __init__(self, ctx, input_size, num_embed, batch_size, update_period, opt_params):
        self._build_model(ctx, input_size, num_embed, batch_size, opt_params)
        self._examples_passed = 0
        self.update_period = update_period
        self.input_size = input_size
        self.num_embed = num_embed
        self.batch_size = batch_size

    def _build_model(self, ctx, input_size, num_embed, batch_size, opt_params):
        input_weight = mx.sym.Variable('input_weight')
        output_weight = mx.sym.Variable('output_weight')
        middle_word = mx.sym.Variable('middle_word')
        ctx_word = mx.sym.Variable('ctx_word')
        label = mx.sym.Variable('label')

        input_embed = mx.sym.Embedding(data=middle_word,
            weight=input_weight,
            input_dim=input_size,
            output_dim=num_embed,
            name='input_embed')

        output_embed = mx.sym.Embedding(data=ctx_word,
            weight=output_weight,
            input_dim=input_size,
            output_dim=num_embed,
            name='output_embed')

        els_prod = input_embed * output_embed
        slice_layers = mx.sym.SliceChannel(data=els_prod,
            num_outputs=num_embed,
            name='slice_layer')
        slice_layers = [slice_layers[i] for i in range(num_embed)]
        els_sum = mx.sym.ElementWiseSum(*slice_layers,
            num_args=num_embed,
            name='elementwise_sum')

        sigmoid = mx.symbol.Activation(name='sigmoid',
            data=els_sum,
            act_type='sigmoid')

        model = mx.symbol.LogisticRegressionOutput(data=sigmoid, label=label, name='cross_entropy')

        arg_names = model.list_arguments()
        input_shapes = {}

        for name in arg_names:
            if name.endswith('word'):
                input_shapes[name] = (batch_size,)
            elif name.endswith('label'):
                input_shapes[name] = (batch_size,)

        arg_shape, out_shape, aux_shape = model.infer_shape(**input_shapes)
        arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
        for i in range(len(arg_shape)):
            arg_arrays[i][:] = mx.rnd.uniform(-0.1, 0.1, arg_shape[i])

        arg_grad = {}
        for shape, name in zip(arg_shape, arg_names):
            if is_param_name(name):
                arg_grad[name] = mx.nd.zeros(shape, ctx)
        arg_dict = dict(zip(arg_names, arg_arrays))

        self.model = model
        self.label = label
        self.arg_arrays = arg_arrays
        self.arg_dict = arg_dict
        self.arg_grad = arg_grad

        grad_req = {}
        for name in arg_names:
            if name.endswith('word') or name.endswith('label'):
                grad_req[name] = 'null'
            else:
                grad_req[name] = 'write'

        self.embed_exec = model.bind(ctx=ctx,
            args=self.arg_dict,
            args_grad=self.arg_grad,
            grad_req=grad_req)


        params_blocks = []
        for i, name in enumerate(arg_names):
            if is_param_name(name):
                params_blocks.append((i, arg_dict[name], arg_grad[name], name))
        self.params_blocks = params_blocks

        opt = mx.optimizer.create('sgd', **opt_params)
        updater = mx.optimizer.get_updater(opt)
        self.updater = updater


    def fit(self, batch):
        middle_words = map(lambda x: x[0][0], batch)
        context_words = map(lambda x: x[0][1], batch)
        labels = map(lambda x: x[1], batch)
        mx.nd.array(middle_words).copyto(self.arg_dict['middle_word'])
        mx.nd.array(context_words).copyto(self.arg_dict['ctx_word'])
        mx.nd.array(labels).copyto(self.arg_dict['label'])
        self.embed_exec.forward(is_train=True)
        self.embed_exec.backward()

        self._examples_passed += 1
        if self._examples_passed % self.update_period == 0:
            for idx, weight, grad, name in self.params_blocks:
                self.updater(idx, grad, weight)
                grad[:] = 0.0

    def get_embedding_cpu(self):
        embedding = np.zeros((self.input_size, self.num_embed))
        self.arg_dict['input_weight'].copyto(embedding)

        return embedding
