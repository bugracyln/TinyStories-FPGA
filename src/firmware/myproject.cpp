#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/altera/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
class Layer5OutPipeID;
using Layer5OutPipe = sycl::ext::altera::experimental::pipe<Layer5OutPipeID, embedding_8_t, 48>;
class Layer6OutPipeID;
using Layer6OutPipe = sycl::ext::altera::experimental::pipe<Layer6OutPipeID, embedding_9_t, 48>;
class Layer9OutPipeID;
using Layer9OutPipe = sycl::ext::altera::experimental::pipe<Layer9OutPipeID, q_add_16_t, 48>;
class Layer81Cpy1PipeID;
using Layer81Cpy1Pipe = sycl::ext::altera::experimental::pipe<Layer81Cpy1PipeID, q_add_16_t, 48>;
class Layer81Cpy2PipeID;
using Layer81Cpy2Pipe = sycl::ext::altera::experimental::pipe<Layer81Cpy2PipeID, q_add_16_t, 48>;
class Layer11OutPipeID;
using Layer11OutPipe = sycl::ext::altera::experimental::pipe<Layer11OutPipeID, q_bn_1_0_t, 48>;
class Layer82Cpy1PipeID;
using Layer82Cpy1Pipe = sycl::ext::altera::experimental::pipe<Layer82Cpy1PipeID, q_bn_1_0_t, 48>;
class Layer82Cpy2PipeID;
using Layer82Cpy2Pipe = sycl::ext::altera::experimental::pipe<Layer82Cpy2PipeID, q_bn_1_0_t, 48>;
class Layer82Cpy3PipeID;
using Layer82Cpy3Pipe = sycl::ext::altera::experimental::pipe<Layer82Cpy3PipeID, q_bn_1_0_t, 48>;
class Layer13OutPipeID;
using Layer13OutPipe = sycl::ext::altera::experimental::pipe<Layer13OutPipeID, q_mha_0_query_t, 48>;
class Layer16OutPipeID;
using Layer16OutPipe = sycl::ext::altera::experimental::pipe<Layer16OutPipeID, q_mha_0_key_t, 48>;
class Layer19OutPipeID;
using Layer19OutPipe = sycl::ext::altera::experimental::pipe<Layer19OutPipeID, q_mha_0_value_t, 48>;
class Layer21OutPipeID;
using Layer21OutPipe = sycl::ext::altera::experimental::pipe<Layer21OutPipeID, q_mha_0_q_mha_0_QK_t, 48>;
class Layer22OutPipeID;
using Layer22OutPipe = sycl::ext::altera::experimental::pipe<Layer22OutPipeID, q_mha_0_q_softmax_t, 48>;
class Layer24OutPipeID;
using Layer24OutPipe = sycl::ext::altera::experimental::pipe<Layer24OutPipeID, q_mha_0_q_mha_0_aV_t, 48>;
class Layer26OutPipeID;
using Layer26OutPipe = sycl::ext::altera::experimental::pipe<Layer26OutPipeID, q_mha_0_attention_output_t, 48>;
class Layer29OutPipeID;
using Layer29OutPipe = sycl::ext::altera::experimental::pipe<Layer29OutPipeID, q_add_17_t, 48>;
class Layer83Cpy1PipeID;
using Layer83Cpy1Pipe = sycl::ext::altera::experimental::pipe<Layer83Cpy1PipeID, q_add_17_t, 48>;
class Layer83Cpy2PipeID;
using Layer83Cpy2Pipe = sycl::ext::altera::experimental::pipe<Layer83Cpy2PipeID, q_add_17_t, 48>;
class Layer31OutPipeID;
using Layer31OutPipe = sycl::ext::altera::experimental::pipe<Layer31OutPipeID, q_bn_2_0_t, 48>;
class Layer76OutPipeID;
using Layer76OutPipe = sycl::ext::altera::experimental::pipe<Layer76OutPipeID, q_dense_12_t, 48>;
class Layer34OutPipeID;
using Layer34OutPipe = sycl::ext::altera::experimental::pipe<Layer34OutPipeID, q_dense_12_relu_t, 48>;
class Layer77OutPipeID;
using Layer77OutPipe = sycl::ext::altera::experimental::pipe<Layer77OutPipeID, q_dense_13_t, 48>;
class Layer37OutPipeID;
using Layer37OutPipe = sycl::ext::altera::experimental::pipe<Layer37OutPipeID, q_dense_13_relu_t, 48>;
class Layer40OutPipeID;
using Layer40OutPipe = sycl::ext::altera::experimental::pipe<Layer40OutPipeID, q_add_18_t, 48>;
class Layer84Cpy1PipeID;
using Layer84Cpy1Pipe = sycl::ext::altera::experimental::pipe<Layer84Cpy1PipeID, q_add_18_t, 48>;
class Layer84Cpy2PipeID;
using Layer84Cpy2Pipe = sycl::ext::altera::experimental::pipe<Layer84Cpy2PipeID, q_add_18_t, 48>;
class Layer42OutPipeID;
using Layer42OutPipe = sycl::ext::altera::experimental::pipe<Layer42OutPipeID, q_bn_1_1_t, 48>;
class Layer85Cpy1PipeID;
using Layer85Cpy1Pipe = sycl::ext::altera::experimental::pipe<Layer85Cpy1PipeID, q_bn_1_1_t, 48>;
class Layer85Cpy2PipeID;
using Layer85Cpy2Pipe = sycl::ext::altera::experimental::pipe<Layer85Cpy2PipeID, q_bn_1_1_t, 48>;
class Layer85Cpy3PipeID;
using Layer85Cpy3Pipe = sycl::ext::altera::experimental::pipe<Layer85Cpy3PipeID, q_bn_1_1_t, 48>;
class Layer44OutPipeID;
using Layer44OutPipe = sycl::ext::altera::experimental::pipe<Layer44OutPipeID, q_mha_1_query_t, 48>;
class Layer47OutPipeID;
using Layer47OutPipe = sycl::ext::altera::experimental::pipe<Layer47OutPipeID, q_mha_1_key_t, 48>;
class Layer50OutPipeID;
using Layer50OutPipe = sycl::ext::altera::experimental::pipe<Layer50OutPipeID, q_mha_1_value_t, 48>;
class Layer52OutPipeID;
using Layer52OutPipe = sycl::ext::altera::experimental::pipe<Layer52OutPipeID, q_mha_1_q_mha_1_QK_t, 48>;
class Layer53OutPipeID;
using Layer53OutPipe = sycl::ext::altera::experimental::pipe<Layer53OutPipeID, q_mha_1_q_softmax_1_t, 48>;
class Layer55OutPipeID;
using Layer55OutPipe = sycl::ext::altera::experimental::pipe<Layer55OutPipeID, q_mha_1_q_mha_1_aV_t, 48>;
class Layer57OutPipeID;
using Layer57OutPipe = sycl::ext::altera::experimental::pipe<Layer57OutPipeID, q_mha_1_attention_output_t, 48>;
class Layer60OutPipeID;
using Layer60OutPipe = sycl::ext::altera::experimental::pipe<Layer60OutPipeID, q_add_19_t, 48>;
class Layer86Cpy1PipeID;
using Layer86Cpy1Pipe = sycl::ext::altera::experimental::pipe<Layer86Cpy1PipeID, q_add_19_t, 48>;
class Layer86Cpy2PipeID;
using Layer86Cpy2Pipe = sycl::ext::altera::experimental::pipe<Layer86Cpy2PipeID, q_add_19_t, 48>;
class Layer62OutPipeID;
using Layer62OutPipe = sycl::ext::altera::experimental::pipe<Layer62OutPipeID, q_bn_2_1_t, 48>;
class Layer78OutPipeID;
using Layer78OutPipe = sycl::ext::altera::experimental::pipe<Layer78OutPipeID, q_dense_14_t, 48>;
class Layer65OutPipeID;
using Layer65OutPipe = sycl::ext::altera::experimental::pipe<Layer65OutPipeID, q_dense_14_relu_t, 48>;
class Layer79OutPipeID;
using Layer79OutPipe = sycl::ext::altera::experimental::pipe<Layer79OutPipeID, q_dense_15_t, 48>;
class Layer68OutPipeID;
using Layer68OutPipe = sycl::ext::altera::experimental::pipe<Layer68OutPipeID, q_dense_15_relu_t, 48>;
class Layer71OutPipeID;
using Layer71OutPipe = sycl::ext::altera::experimental::pipe<Layer71OutPipeID, q_add_20_t, 48>;
class Layer73OutPipeID;
using Layer73OutPipe = sycl::ext::altera::experimental::pipe<Layer73OutPipeID, q_bn_final_t, 48>;

using token_id_emb_t = typename nnet::array<token_inputs_t::value_type, 1>;
class LoopBackEmbPipeID;
using LoopBackEmbPipe = sycl::ext::altera::experimental::pipe<LoopBackEmbPipeID, token_id_emb_t, 48>;

class LayerLBembOutPipeID;
using LayerLBembOutPipe = sycl::ext::altera::experimental::pipe<LayerLBembOutPipeID, embedding_8_t, 48>;

using token_id_pos_t = typename nnet::array<pos_inputs_t::value_type, 1>;
class LoopBackPosPipeID;
using LoopBackPosPipe = sycl::ext::altera::experimental::pipe<LoopBackPosPipeID, token_id_pos_t, 48>;

class LayerLBposOutPipeID;
using LayerLBposOutPipe = sycl::ext::altera::experimental::pipe<LayerLBposOutPipeID, embedding_9_t, 48>;

class LayerLBOutPipeID;
using LayerLBOutPipe = sycl::ext::altera::experimental::pipe<LayerLBOutPipeID, q_add_16_t, 48>;

class LayerSwitchPipeID;
using LayerSwitchPipe = sycl::ext::altera::experimental::pipe<LayerSwitchPipeID, q_add_16_t, 48>;

class LayerVocabOutPipeID;
using LayerVocabOutPipe = sycl::ext::altera::experimental::pipe<LayerVocabOutPipeID, result_t, 48>;


const uint32_t MAX_INVOC = 10;
using ts_properties = decltype(sycl::ext::oneapi::experimental::properties{
    sycl::ext::altera::experimental::invocation_capacity<MAX_INVOC>,
    sycl::ext::altera::experimental::response_capacity<MAX_INVOC>});


using sycl::ext::altera::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences
    task_sequence<nnet::embedding_stream<TokenInputsPipe, Layer5OutPipe, config5>,ts_properties> embedding_8;
    task_sequence<nnet::embedding_stream<PosInputsPipe, Layer6OutPipe, config6>,ts_properties> embedding_9;
    task_sequence<nnet::add_stream<Layer5OutPipe, Layer6OutPipe, Layer9OutPipe, config9>,ts_properties> q_add_16;

    task_sequence<nnet::embedding_stream<LoopBackEmbPipe, LayerLBembOutPipe, configEmbLB>,ts_properties> embedding_emb_lb;
    task_sequence<nnet::embedding_stream<LoopBackPosPipe, LayerLBposOutPipe, configPosLB>,ts_properties> embedding_pos_lb;
    task_sequence<nnet::add_stream<LayerLBembOutPipe, LayerLBposOutPipe, LayerLBOutPipe, configAddLB>,ts_properties> q_add_16_lb;

    task_sequence<nnet::input_switch<Layer9OutPipe, LayerLBOutPipe, LayerSwitchPipe, configInSW>,ts_properties> input_sw;
    task_sequence<nnet::clone_stream<LayerSwitchPipe, Layer81Cpy1Pipe, Layer81Cpy2Pipe, 1536>,ts_properties> clone_q_add_16;

    task_sequence<nnet::normalize_stream<Layer81Cpy1Pipe, Layer11OutPipe, config11>,ts_properties> q_bn_1_0;
    task_sequence<nnet::clone_stream<Layer11OutPipe, Layer82Cpy1Pipe, Layer82Cpy2Pipe, Layer82Cpy3Pipe, 1536>,ts_properties> clone_q_bn_1_0;
    task_sequence<nnet::einsum_dense_stream<Layer82Cpy1Pipe, Layer13OutPipe, config13>,ts_properties> q_mha_0_query;
    task_sequence<nnet::einsum_dense_stream<Layer82Cpy2Pipe, Layer16OutPipe, config16>,ts_properties> q_mha_0_key;
    task_sequence<nnet::einsum_dense_stream<Layer82Cpy3Pipe, Layer19OutPipe, config19>,ts_properties> q_mha_0_value;
    task_sequence<nnet::causal_einsum<Layer13OutPipe, Layer16OutPipe, Layer21OutPipe, config21>,ts_properties> q_mha_0_q_mha_0_QK;
    task_sequence<nnet::softmax_stream<Layer21OutPipe, Layer22OutPipe, softmax_config22>,ts_properties> q_mha_0_q_softmax;
    task_sequence<nnet::causal_einsum<Layer22OutPipe, Layer19OutPipe, Layer24OutPipe, config24>,ts_properties> q_mha_0_q_mha_0_aV;
    task_sequence<nnet::einsum_dense_stream<Layer24OutPipe, Layer26OutPipe, config26>,ts_properties> q_mha_0_attention_output;
    task_sequence<nnet::add_stream<Layer81Cpy2Pipe, Layer26OutPipe, Layer29OutPipe, config29>,ts_properties> q_add_17;
    task_sequence<nnet::clone_stream<Layer29OutPipe, Layer83Cpy1Pipe, Layer83Cpy2Pipe, 1536>,ts_properties> clone_q_add_17;
    task_sequence<nnet::normalize_stream<Layer83Cpy1Pipe, Layer31OutPipe, config31>,ts_properties> q_bn_2_0;
    task_sequence<nnet::conv_1d_cl_stream<Layer31OutPipe, Layer76OutPipe, config76>,ts_properties> q_dense_12;
    task_sequence<nnet::relu_stream<Layer76OutPipe, Layer34OutPipe, relu_config34>,ts_properties> q_dense_12_relu;
    task_sequence<nnet::conv_1d_cl_stream<Layer34OutPipe, Layer77OutPipe, config77>,ts_properties> q_dense_13;
    task_sequence<nnet::relu_stream<Layer77OutPipe, Layer37OutPipe, relu_config37>,ts_properties> q_dense_13_relu;
    task_sequence<nnet::add_stream<Layer83Cpy2Pipe, Layer37OutPipe, Layer40OutPipe, config40>,ts_properties> q_add_18;
    task_sequence<nnet::clone_stream<Layer40OutPipe, Layer84Cpy1Pipe, Layer84Cpy2Pipe, 1536>,ts_properties> clone_q_add_18;
    task_sequence<nnet::normalize_stream<Layer84Cpy1Pipe, Layer42OutPipe, config42>,ts_properties> q_bn_1_1;
    task_sequence<nnet::clone_stream<Layer42OutPipe, Layer85Cpy1Pipe, Layer85Cpy2Pipe, Layer85Cpy3Pipe, 1536>,ts_properties> clone_q_bn_1_1;
    task_sequence<nnet::einsum_dense_stream<Layer85Cpy1Pipe, Layer44OutPipe, config44>,ts_properties> q_mha_1_query;
    task_sequence<nnet::einsum_dense_stream<Layer85Cpy2Pipe, Layer47OutPipe, config47>,ts_properties> q_mha_1_key;
    task_sequence<nnet::einsum_dense_stream<Layer85Cpy3Pipe, Layer50OutPipe, config50>,ts_properties> q_mha_1_value;
    task_sequence<nnet::causal_einsum<Layer44OutPipe, Layer47OutPipe, Layer52OutPipe, config52>,ts_properties> q_mha_1_q_mha_1_QK;
    task_sequence<nnet::softmax_stream<Layer52OutPipe, Layer53OutPipe, softmax_config53>,ts_properties> q_mha_1_q_softmax_1;
    task_sequence<nnet::causal_einsum<Layer53OutPipe, Layer50OutPipe, Layer55OutPipe, config55>,ts_properties> q_mha_1_q_mha_1_aV;
    task_sequence<nnet::einsum_dense_stream<Layer55OutPipe, Layer57OutPipe, config57>,ts_properties> q_mha_1_attention_output;
    task_sequence<nnet::add_stream<Layer84Cpy2Pipe, Layer57OutPipe, Layer60OutPipe, config60>,ts_properties> q_add_19;
    task_sequence<nnet::clone_stream<Layer60OutPipe, Layer86Cpy1Pipe, Layer86Cpy2Pipe, 1536>,ts_properties> clone_q_add_19;
    task_sequence<nnet::normalize_stream<Layer86Cpy1Pipe, Layer62OutPipe, config62>,ts_properties> q_bn_2_1;
    task_sequence<nnet::conv_1d_cl_stream<Layer62OutPipe, Layer78OutPipe, config78>,ts_properties> q_dense_14;
    task_sequence<nnet::relu_stream<Layer78OutPipe, Layer65OutPipe, relu_config65>,ts_properties> q_dense_14_relu;
    task_sequence<nnet::conv_1d_cl_stream<Layer65OutPipe, Layer79OutPipe, config79>,ts_properties> q_dense_15;
    task_sequence<nnet::relu_stream<Layer79OutPipe, Layer68OutPipe, relu_config68>,ts_properties> q_dense_15_relu;
    task_sequence<nnet::add_stream<Layer86Cpy2Pipe, Layer68OutPipe, Layer71OutPipe, config71>,ts_properties> q_add_20;
    task_sequence<nnet::normalize_stream<Layer71OutPipe, Layer73OutPipe, config73>,ts_properties> q_bn_final;

    task_sequence<nnet::conv_1d_cl_stream<Layer73OutPipe, LayerVocabOutPipe, config80>,ts_properties> lm_head;
    task_sequence<nnet::output_switch<LayerVocabOutPipe, LoopBackEmbPipe, LoopBackPosPipe, Layer80OutPipe, configOutSW>,ts_properties> output_sw;

    // hls-fpga-machine-learning insert layers
    //unsigned num_iterations = 10;
    //for(unsigned i = 0; i < num_iterations; i++){
    embedding_8.async();
    embedding_9.async();
    q_add_16.async();

    embedding_emb_lb.async();
    embedding_pos_lb.async();
    q_add_16_lb.async();
    
    input_sw.async();

    clone_q_add_16.async();
    q_bn_1_0.async();
    clone_q_bn_1_0.async();
    q_mha_0_query.async();
    q_mha_0_key.async();
    q_mha_0_value.async();
    q_mha_0_q_mha_0_QK.async();
    q_mha_0_q_softmax.async();
    q_mha_0_q_mha_0_aV.async();
    q_mha_0_attention_output.async();
    q_add_17.async();
    clone_q_add_17.async();
    q_bn_2_0.async();
    q_dense_12.async();
    q_dense_12_relu.async();
    q_dense_13.async();
    q_dense_13_relu.async();
    q_add_18.async();
    clone_q_add_18.async();
    q_bn_1_1.async();
    clone_q_bn_1_1.async();
    q_mha_1_query.async();
    q_mha_1_key.async();
    q_mha_1_value.async();
    q_mha_1_q_mha_1_QK.async();
    q_mha_1_q_softmax_1.async();
    q_mha_1_q_mha_1_aV.async();
    q_mha_1_attention_output.async();
    q_add_19.async();
    clone_q_add_19.async();
    q_bn_2_1.async();
    q_dense_14.async();
    q_dense_14_relu.async();
    q_dense_15.async();
    q_dense_15_relu.async();
    q_add_20.async();
    q_bn_final.async();
    lm_head.async();

    output_sw.async();
    //}

    // hls-fpga-machine-learning return
}
