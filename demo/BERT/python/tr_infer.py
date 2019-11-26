#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import ctypes
import numpy as np
import tokenization
import tensorrt as trt
import data_processing as dp
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
class Model(object):

    def __init__(self):
        vocab_file = '../vocab.txt'
        bert_engine = '../bert_base_384.engine'
        paragraph_text =  "today is a good day to play."
        question_text = "today is what?"

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
        self.max_query_length = 64
        # When splitting up a long document into chunks, how much stride to take between chunks.
        self.doc_stride = 128
        # The maximum total input sequence length after WordPiece tokenization.
        # Sequences longer than this will be truncated, and sequences shorter
        self.max_seq_length = 384
        # Extract tokecs from the paragraph
        # Import necessary plugins for BERT TensorRT
        ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libcommon.so", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libbert_plugins.so", mode=ctypes.RTLD_GLOBAL)
        f = open(bert_engine, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        f.close()

        input_shape = (1, self.max_seq_length)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

        # Allocate device memory for inputs.
        self.d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

        # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        for binding in range(3):
            self.context.set_binding_shape(binding, input_shape)
        assert self.context.all_binding_shapes_specified

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        self.h_output = cuda.pagelocked_empty(tuple(self.context.get_binding_shape(3)), dtype=np.float32)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

    def question_features(self, paragraph_text, question):
        doc_tokens = dp.convert_doc_tokens(paragraph_text)
        # Extract features from the paragraph and question
        return dp.convert_examples_to_features(doc_tokens, question, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length), doc_tokens

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.

    def inference(self, paragraph_text, question):
        all_features, doc_tokens = self.question_features(paragraph_text, question)
        print("\nRunning Inference...")
        eval_start_time = time.time()
        print(len(all_features))
        best_prediction = ''
        best_p = -1
        best_np = 100

        for feature in all_features:
            # Copy inputs
            cuda.memcpy_htod_async(self.d_inputs[0], feature["input_ids"], self.stream)
            cuda.memcpy_htod_async(self.d_inputs[1], feature["segment_ids"], self.stream)
            cuda.memcpy_htod_async(self.d_inputs[2], feature["input_mask"], self.stream)
            # Run inference
            self.context.execute_async_v2(bindings=[int(d_inp) for d_inp in self.d_inputs] + [int(self.d_output)], stream_handle=self.stream.handle)
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            # Synchronize the stream
            self.stream.synchronize()

            eval_time_elapsed = time.time() - eval_start_time

            # print("------------------------")
            # print("Running inference in {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
            # print("------------------------")


            for index, batch in enumerate(self.h_output):

                # Data Post-processing
                start_logits = batch[:, 0]
                end_logits = batch[:, 1]

                # Total number of n-best predictions to generate in the nbest_predictions.json output file
                n_best_size = 20

                # The maximum length of an answer that can be generated. This is needed
                # because the start and end predictions are not conditioned on one another
                max_answer_length = 30

                prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, feature,
                        start_logits, end_logits, n_best_size, max_answer_length)
                if prediction == '' and nbest_json[0]['probability'] < best_np:
                    best_np = nbest_json[0]['probability']
                elif prediction != '' and nbest_json[0]['probability'] > best_p:
                    best_p = nbest_json[0]['probability']
                    best_prediction = prediction

        print("Answer: '{}'".format(best_prediction))
        final_p = best_p if best_prediction != '' else best_np
        print("With probability: {:.3f}".format(final_p * 100.0))
        output = {}
        output['result']= best_prediction
        output['p'] = final_p
        return output

if __name__=="__main__":
    vocab_file = '../vocab.txt'
    bert_engine = '../bert_base_384.engine'
    paragraph_text =  "In finance, an option is a contract which gives the buyer the owner or holder of the option the right, but not the obligation, to buy or sell an underlying asset or instrument at a specified strike price prior to or on a specified date, depending on the form of the option. The strike price may be set by reference to the spot price (market price) of the underlying security or commodity on the day an option is taken out, or it may be fixed at a discount or at a premium. The seller has the corresponding obligation to fulfill the transaction to sell or buy if the buyer (owner'exercises' the option. An option that conveys to the owner the right to buy at a specific price is referred to as a call; an option that conveys the right of the owner to sell at a specific price is referred to as a put. Both are commonly traded, but the call option is more frequently discussed. The seller may grant an option to a buyer as part of another transaction, such as a share issue or as part of an employee incentive scheme, otherwise a buyer would pay a premium to the seller for the option. A call option would normally be exercised only when the strike price is below the market value of the underlying asset, while a put option would normally be exercised only when the strike price is above the market value. When an option is exercised, the cost to the buyer of the asset acquired is the strike price plus the premium, if any. When the option expiration date passes without the option being exercised, the option expires and the buyer would forfeit the premium to the seller. In any case, the premium is income to the seller, and normally a capital loss to the buyer. The owner of an option may on-sell the option to a third party in a secondary market, in either an over-the-counter transaction or on an options exchange, depending on the option. The market price of an American-style option normally closely follows that of the underlying stock being the difference between the market price of the stock and the strike price of the option. The actual market price of the option may vary depending on a number of factors, such as a significant option holder may need to sell the option as the expiry date is approaching and does not have the financial resources to exercise the option, or a buyer in the market is trying to amass a large option holding. The ownership of an option does not generally entitle the holder to any rights associated with the underlying asset, such as voting rights or any income from the underlying asset, such as a dividend."
    f = open('plan_doc.txt','r')
    paragraph_text = f.read()
    question_text = "where can I find my account statment?"
    m = Model()
    m.inference(paragraph_text, question_text)
