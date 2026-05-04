void softmax_naive(
    const float* logits,
    float* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len,
    bool casual);

void softmax_2_pass(
    const float* logits,
    float* output,
    int batch_size,
    int num_heads,
    int query_len,
    int key_len,
    bool casual    
);
