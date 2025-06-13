#!/usr/bin/env python3
import sys
import tensorflow as tf

print("🔧 Forcing CPU for batch matrix operations...")

# Save original
_orig_matmul = tf.matmul
_orig_batch_matmul = None
try:
    from tensorflow.python.ops import gen_math_ops
    _orig_batch_matmul = gen_math_ops.batch_mat_mul_v2
except:
    pass

def cpu_batch_matmul(a, b, **kwargs):
    # Force to CPU
    with tf.device("/cpu:0"):
        return _orig_matmul(a, b, **kwargs)

# Override both matmul and batch_mat_mul
tf.matmul = cpu_batch_matmul

# Override gen_math_ops batch operations
try:
    from tensorflow.python.ops import gen_math_ops
    
    def cpu_batch_mat_mul(x, y, adj_x=False, adj_y=False, name=None):
        with tf.device("/cpu:0"):
            if _orig_batch_matmul:
                return _orig_batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y, name=name)
            else:
                # Fallback to regular matmul
                return tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y, name=name)
    
    gen_math_ops.batch_mat_mul = cpu_batch_mat_mul
    gen_math_ops.batch_mat_mul_v2 = cpu_batch_mat_mul
    print("  ✓ Forced CPU for gen_math_ops batch operations")
except:
    pass

print("✅ CPU fallback enabled for ALL matrix operations")

from tensor2tensor.bin import t2t_trainer
t2t_trainer.main(argv=sys.argv[1:])
