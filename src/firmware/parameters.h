#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert softmax tables
#include "nnet_utils/activation_tables/q_mha_0_q_softmax_exp_table.h"
#include "nnet_utils/activation_tables/q_mha_0_q_softmax_inv_table.h"
#include "nnet_utils/activation_tables/q_mha_1_q_softmax_1_exp_table.h"
#include "nnet_utils/activation_tables/q_mha_1_q_softmax_1_inv_table.h"

// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_causal_einsum.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_einsum.h"
#include "nnet_utils/nnet_einsum_dense_stream.h"
#include "nnet_utils/nnet_embed.h"
#include "nnet_utils/nnet_embed_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_stream.h"
#include "nnet_utils/nnet_dma_helpers.h"
#include "nnet_utils/nnet_data_movement.h"

//Transpose weights function (REMOVE LATER)
template< class arr_T, unsigned orig_rows, unsigned orig_cols> 
constexpr auto tpose(const arr_T &arr) {
	arr_T result;
	
	for(unsigned r = 0; r < orig_rows; r++){
		for(unsigned c = 0; c < orig_cols; c++){
			result[c * orig_rows + r] = arr[r * orig_cols + c]; 	
		}
	}
	return result;
}

// hls-fpga-machine-learning insert layer-config

struct configInSW {
    static const unsigned n_prefill = 32;
    static const unsigned n_generate = 16;
};

struct configOutSW {
    static const unsigned n_prefill = 32;
    static const unsigned n_generate = 16;
};

struct configEmbLB : nnet::embed_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned vocab_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    typedef e5_t embeddings_t;
    
    static const unsigned unroll_factor = 32;
    [[intel::fpga_memory, intel::numbanks(unroll_factor), intel::bankwidth(sizeof(embeddings_t::value_type))]] static constexpr embeddings_t embeddings = e5;
};

struct configPosLB : nnet::embed_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned vocab_size = 48;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    typedef e6_t embeddings_t;

    static const unsigned unroll_factor = 32;
    [[intel::fpga_memory, intel::numbanks(unroll_factor), intel::bankwidth(sizeof(embeddings_t::value_type))]] static constexpr embeddings_t embeddings = e6;
};

struct configAddLB : nnet::merge_config {
    static const unsigned n_elem = 16*32;
    static const unsigned reuse_factor = 4;
};

struct config5 : nnet::embed_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 32;
    static const unsigned vocab_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    typedef e5_t embeddings_t;

    static const unsigned unroll_factor = 32;
    [[intel::fpga_memory, intel::numbanks(unroll_factor), intel::bankwidth(sizeof(embeddings_t::value_type))]] static constexpr embeddings_t embeddings = e5;
};

struct config6 : nnet::embed_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 32;
    static const unsigned vocab_size = 48;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    typedef e6_t embeddings_t;

    static const unsigned unroll_factor = 32;
    [[intel::fpga_memory, intel::numbanks(unroll_factor), intel::bankwidth(sizeof(embeddings_t::value_type))]] static constexpr embeddings_t embeddings = e6;
};

struct config9 : nnet::merge_config {
    static const unsigned n_elem = 32*32;
    static const unsigned reuse_factor = 4;
};

struct config11 : nnet::batchnorm_config {
    static constexpr unsigned n_in = 48*32;
    static constexpr unsigned n_filt = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr bool store_weights_in_bram = false;
    typedef b11_t bias_t;
    typedef s11_t scale_t;
    static constexpr bias_t bias = b11;
    static constexpr scale_t scale = s11;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config13_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 84;
    static constexpr unsigned n_nonzeros = 940;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_0_query_accum_t accum_t;
    typedef b13_t bias_t;
    typedef w13_t weight_t;

    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w13);
    static constexpr bias_t biases = b13;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config13 {

    typedef q_mha_0_query_accum_t accum_t;
    typedef w13_t weight_t;
    typedef b13_t bias_t;

    //static constexpr const auto *weights = w13.data();
    //static constexpr const auto *biases = b13.data();
    
    static constexpr w13_t weights = w13;
    static constexpr b13_t biases = b13;


    static constexpr bool opt_dense = 0;
    static constexpr unsigned n_head = 1;

    typedef config13_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};

struct config16_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 56;
    static constexpr unsigned n_nonzeros = 968;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_0_key_accum_t accum_t;
    typedef b16_t bias_t;
    typedef w16_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w16);
    static constexpr bias_t biases = b16; 

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config16 {

    typedef q_mha_0_key_accum_t accum_t;
    typedef w16_t weight_t;
    typedef b16_t bias_t;

    //static constexpr const auto *weights = w16.data();
    //static constexpr const auto *biases = b16.data();
    
    static constexpr w16_t weights = w16;
    static constexpr b16_t biases = b16;

    static constexpr bool opt_dense = 0;
    static constexpr unsigned n_head = 1;

    typedef config16_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};

struct config19_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 24;
    static constexpr unsigned n_nonzeros = 1000;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_0_value_accum_t accum_t;
    typedef b19_t bias_t;
    typedef w19_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w19);
    static constexpr bias_t biases = b19;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config19 {

    typedef q_mha_0_value_accum_t accum_t;
    typedef w19_t weight_t;
    typedef b19_t bias_t;

    static constexpr w19_t weights = w19;
    static constexpr b19_t biases = b19;

    static constexpr bool opt_dense = 0;
    static constexpr unsigned n_head = 1;

    typedef config19_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};


struct config21 {

    typedef q_mha_0_q_mha_0_QK_accum_t accum_t;

    // Context Length
    static const unsigned n_ctx = 48;

    // BRAM Contraction Axis
    static const bool contract_dim = 0;

    // Square Root of KEY_DIM
    static constexpr ac_fixed<4,1,true,AC_RND,AC_SAT> sqrt_dk = 5.656854249492381;

    // Layer Sizes
    static const unsigned n_free0 = 1;
    static const unsigned n_free1 = 1;
    static const unsigned n_contract = 32;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    static const unsigned multiplier_limit = 8;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config22 : nnet::activ_config {
    static constexpr unsigned n_in = 2304;
    static constexpr unsigned exp_table_size = 32;
    static constexpr unsigned inv_table_size = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef q_mha_0_q_softmax_exp_table_t exp_table_t;
    typedef q_mha_0_q_softmax_inv_table_t inv_table_t;
    typedef q_mha_0_q_softmax_inv_inp_t inv_inp_t;
    typedef q_mha_0_q_softmax_inp_norm_t inp_norm_t;

    //static constexpr const exp_table_t *exp_table = &q_mha_0_q_softmax_exp_table[0];
    //static constexpr const inv_table_t *invert_table = &q_mha_0_q_softmax_inv_table[0];
    
    static constexpr q_mha_0_q_softmax_exp_table_arr_t exp_table = q_mha_0_q_softmax_exp_table;
    static constexpr q_mha_0_q_softmax_inv_table_arr_t invert_table = q_mha_0_q_softmax_inv_table;

};


struct config24 {

    typedef q_mha_0_q_mha_0_aV_accum_t accum_t;

    // Context Length
    static const unsigned n_ctx = 48;

    // BRAM Contraction Axis
    static const bool contract_dim = 1;

    // Square Root of KEY_DIM
    static constexpr unsigned sqrt_dk = 1;

    // Layer Sizes
    static const unsigned n_free0 = 48;
    static const unsigned n_free1 = 32;
    static const unsigned n_contract = 1;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    static const unsigned multiplier_limit = 384;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config26_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 8;
    static constexpr unsigned n_nonzeros = 1016;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_0_attention_output_accum_t accum_t;
    typedef b26_t bias_t;
    typedef w26_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w26);
    static constexpr bias_t biases = b26;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config26 {

    typedef q_mha_0_attention_output_accum_t accum_t;
    typedef w26_t weight_t;
    typedef b26_t bias_t;

    static constexpr w26_t weights = w26;
    static constexpr b26_t biases = b26;

    static constexpr bool opt_dense = 1;
    static constexpr unsigned n_head = 1;

    typedef config26_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};

struct config29 : nnet::merge_config {
    static const unsigned n_elem = 48*32;
    static const unsigned reuse_factor = 4;
};

struct config31 : nnet::batchnorm_config {
    static constexpr unsigned n_in = 48*32;
    static constexpr unsigned n_filt = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr bool store_weights_in_bram = false;
    typedef b31_t bias_t;
    typedef s31_t scale_t;
    static constexpr bias_t bias = b31;
    static constexpr scale_t scale = s31;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config76_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 64;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = n_in;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_12_accum_t accum_t;
    typedef b76_t bias_t;
    typedef w76_t weight_t;
    
    static constexpr bool conv = true;
 
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config76 : nnet::conv1d_config {
    static const unsigned in_width = 48;
    static const unsigned n_chan = 32;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 64;
    static const unsigned out_width = 48;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 4;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_12_accum_t accum_t;
    typedef b76_t bias_t;
    typedef w76_t weight_t;
    typedef config76_mult mult_config;

    [[intel::fpga_memory, intel::numbanks(n_chan), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = w76;
    static constexpr bias_t biases = b76;
};

struct relu_config34 : nnet::activ_config {
    static constexpr unsigned n_in = 3072;
    static constexpr unsigned table_size = 131072;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    typedef q_dense_12_relu_table_t table_t;
};

struct config77_mult : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 32;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = n_in;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_13_accum_t accum_t;
    typedef b77_t bias_t;
    typedef w77_t weight_t;
    
    static constexpr bool conv = true;
 
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config77 : nnet::conv1d_config {
    static const unsigned in_width = 48;
    static const unsigned n_chan = 64;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 32;
    static const unsigned out_width = 48;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 4;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_13_accum_t accum_t;
    typedef b77_t bias_t;
    typedef w77_t weight_t;
    typedef config77_mult mult_config;

    [[intel::fpga_memory, intel::numbanks(n_chan), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = w77;
    static constexpr bias_t biases = b77;
};

struct relu_config37 : nnet::activ_config {
    static constexpr unsigned n_in = 1536;
    static constexpr unsigned table_size = 524288;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    typedef q_dense_13_relu_table_t table_t;
};

struct config40 : nnet::merge_config {
    static const unsigned n_elem = 48*32;
    static const unsigned reuse_factor = 4;
};

struct config42 : nnet::batchnorm_config {
    static constexpr unsigned n_in = 48*32;
    static constexpr unsigned n_filt = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr bool store_weights_in_bram = false;
    typedef b42_t bias_t;
    typedef s42_t scale_t;
    static constexpr bias_t bias = b42;
    static constexpr scale_t scale = s42;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config44_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 328;
    static constexpr unsigned n_nonzeros = 696;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_1_query_accum_t accum_t;
    typedef b44_t bias_t;
    typedef w44_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w44);
    static constexpr bias_t biases = b44;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config44 {

    typedef q_mha_1_query_accum_t accum_t;
    typedef w44_t weight_t;
    typedef b44_t bias_t;

    static constexpr w44_t weights = w44;
    static constexpr b44_t biases = b44;

    static constexpr bool opt_dense = 0;
    static constexpr unsigned n_head = 1;

    typedef config44_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};

struct config47_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 248;
    static constexpr unsigned n_nonzeros = 776;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_1_key_accum_t accum_t;
    typedef b47_t bias_t;
    typedef w47_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w47);
    static constexpr bias_t biases = b47;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config47 {

    typedef q_mha_1_key_accum_t accum_t;
    typedef w47_t weight_t;
    typedef b47_t bias_t;

    static constexpr w47_t weights = w47;
    static constexpr b47_t biases = b47;

    static constexpr bool opt_dense = 0;
    static constexpr unsigned n_head = 1;

    typedef config47_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};

struct config50_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 18;
    static constexpr unsigned n_nonzeros = 1006;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_1_value_accum_t accum_t;
    typedef b50_t bias_t;
    typedef w50_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w50);
    static constexpr bias_t biases = b50;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config50 {

    typedef q_mha_1_value_accum_t accum_t;
    typedef w50_t weight_t;
    typedef b50_t bias_t;

    static constexpr w50_t weights = w50;
    static constexpr b50_t biases = b50;

    static constexpr bool opt_dense = 0;
    static constexpr unsigned n_head = 1;

    typedef config50_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};


struct config52 {

    typedef q_mha_1_q_mha_1_QK_accum_t accum_t;

    // Context Length
    static const unsigned n_ctx = 48;

    // BRAM Contraction Axis
    static const bool contract_dim = 0;

    // Square Root of KEY_DIM
    static constexpr ac_fixed<2,0,true,AC_RND,AC_SAT> sqrt_dk = 5.656854249492381;

    // Layer Sizes
    static const unsigned n_free0 = 1;
    static const unsigned n_free1 = 1;
    static const unsigned n_contract = 32;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    static const unsigned multiplier_limit = 8;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config53 : nnet::activ_config {
    static constexpr unsigned n_in = 2304;
    static constexpr unsigned exp_table_size = 32;
    static constexpr unsigned inv_table_size = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef q_mha_1_q_softmax_1_exp_table_t exp_table_t;
    typedef q_mha_1_q_softmax_1_inv_table_t inv_table_t;
    typedef q_mha_1_q_softmax_1_inv_inp_t inv_inp_t;
    typedef q_mha_1_q_softmax_1_inp_norm_t inp_norm_t;

    static constexpr q_mha_1_q_softmax_1_exp_table_arr_t exp_table = q_mha_1_q_softmax_1_exp_table;
    static constexpr q_mha_1_q_softmax_1_inv_table_arr_t invert_table = q_mha_1_q_softmax_1_inv_table;
};


struct config55 {

    typedef q_mha_1_q_mha_1_aV_accum_t accum_t;

    // Context Length
    static const unsigned n_ctx = 48;

    // BRAM Contraction Axis
    static const bool contract_dim = 1;

    // Square Root of KEY_DIM
    static constexpr unsigned sqrt_dk = 1;

    // Layer Sizes
    static const unsigned n_free0 = 48;
    static const unsigned n_free1 = 32;
    static const unsigned n_contract = 1;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4;
    static const unsigned multiplier_limit = 384;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config57_dense : nnet::dense_config {
    static constexpr unsigned n_in = 32;
    static constexpr unsigned n_out = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned n_zeros = 28;
    static constexpr unsigned n_nonzeros = 996;
    static constexpr bool store_weights_in_bram = false;

    static constexpr unsigned rf_pad = 0;
    static constexpr unsigned bf_pad = 0;

    static constexpr unsigned reuse_factor = n_in;
    static constexpr unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static constexpr unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static constexpr unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static constexpr unsigned block_factor_rounded = block_factor + bf_pad;
    static constexpr unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static constexpr unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static constexpr unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_mha_1_attention_output_accum_t accum_t;
    typedef b57_t bias_t;
    typedef w57_t weight_t;
    
    static constexpr bool conv = true;
    [[intel::fpga_memory, intel::numbanks(n_in), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = tpose<weight_t,32,32>(w57);
    static constexpr bias_t biases = b57;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



struct config57 {

    typedef q_mha_1_attention_output_accum_t accum_t;
    typedef w57_t weight_t;
    typedef b57_t bias_t;

    static constexpr w57_t weights = w57;
    static constexpr b57_t biases = b57;

    static constexpr bool opt_dense = 1;
    static constexpr unsigned n_head = 1;

    typedef config57_dense dense_conf;

    // Layer Sizes
    static constexpr unsigned n_free_data = 48;
    static constexpr unsigned n_free_kernel = 32;
    static constexpr unsigned n_contract = 32;
    static constexpr unsigned n_inplace = 1;

    // Resource reuse info
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr unsigned parallelization_factor = 48; // Only useful when n_inplace > 1
};

struct config60 : nnet::merge_config {
    static const unsigned n_elem = 48*32;
    static const unsigned reuse_factor = 4;
};

struct config62 : nnet::batchnorm_config {
    static constexpr unsigned n_in = 48*32;
    static constexpr unsigned n_filt = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr bool store_weights_in_bram = false;
    typedef b62_t bias_t;
    typedef s62_t scale_t;
    static constexpr bias_t bias = b62;
    static constexpr scale_t scale = s62;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config78_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 64;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = n_in;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_14_accum_t accum_t;
    typedef b78_t bias_t;
    typedef w78_t weight_t;
    
    static constexpr bool conv = true;
 
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config78 : nnet::conv1d_config {
    static const unsigned in_width = 48;
    static const unsigned n_chan = 32;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 64;
    static const unsigned out_width = 48;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 4;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_14_accum_t accum_t;
    typedef b78_t bias_t;
    typedef w78_t weight_t;
    typedef config78_mult mult_config;

    [[intel::fpga_memory, intel::numbanks(n_chan), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = w78;
    static constexpr bias_t biases = b78;
};

struct relu_config65 : nnet::activ_config {
    static constexpr unsigned n_in = 3072;
    static constexpr unsigned table_size = 262144;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    typedef q_dense_14_relu_table_t table_t;
};

struct config79_mult : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 32;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = n_in;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef q_dense_15_accum_t accum_t;
    typedef b79_t bias_t;
    typedef w79_t weight_t;
    
    static constexpr bool conv = true;
 
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config79 : nnet::conv1d_config {
    static const unsigned in_width = 48;
    static const unsigned n_chan = 64;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 32;
    static const unsigned out_width = 48;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 4;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef q_dense_15_accum_t accum_t;
    typedef b79_t bias_t;
    typedef w79_t weight_t;
    typedef config79_mult mult_config;

    [[intel::fpga_memory, intel::numbanks(n_chan), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = w79;
    static constexpr bias_t biases = b79;
};

struct relu_config68 : nnet::activ_config {
    static constexpr unsigned n_in = 1536;
    static constexpr unsigned table_size = 524288;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    typedef q_dense_15_relu_table_t table_t;
};

struct config71 : nnet::merge_config {
    static const unsigned n_elem = 48*32;
    static const unsigned reuse_factor = 4;
};

struct config73 : nnet::batchnorm_config {
    static constexpr unsigned n_in = 48*32;
    static constexpr unsigned n_filt = 32;
    static constexpr unsigned io_type = nnet::io_stream;
    static constexpr unsigned reuse_factor = 4;
    static constexpr bool store_weights_in_bram = false;
    typedef b73_t bias_t;
    typedef s73_t scale_t;
    static constexpr bias_t bias = b73;
    static constexpr scale_t scale = s73;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config80_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 1024;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = n_in;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef lm_head_accum_t accum_t;
    typedef b80_t bias_t;
    typedef w80_t weight_t;
    
    static constexpr bool conv = true;
 
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config80 : nnet::conv1d_config {
    static const unsigned in_width = 48;
    static const unsigned n_chan = 32;

    static const unsigned filt_width = 1;
    static const unsigned impl_filt_width = 1;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 1024;
    static const unsigned out_width = 48;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 4;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::im2col;

    typedef lm_head_accum_t accum_t;
    typedef b80_t bias_t;
    typedef w80_t weight_t;
    typedef config80_mult mult_config;

    [[intel::fpga_memory, intel::numbanks(n_chan), intel::bankwidth(sizeof(weight_t::value_type))]] static constexpr weight_t weights = w80;
    static constexpr bias_t biases = b80;
};


#endif
