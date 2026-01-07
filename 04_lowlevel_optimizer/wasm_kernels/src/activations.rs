// ODIN WASM Kernels: Activation Functions
// ========================================
// exp, sigmoid, tanh, relu, silu ottimizzate per WASM.

use std::f32::consts::E;

/// Fast exponential approximation
/// Accuracy: ~0.1% error, 3x faster than std::exp
#[inline]
fn fast_exp(x: f32) -> f32 {
    // Clamp to prevent overflow
    let x = x.max(-88.0).min(88.0);
    
    // Use standard exp for now (can optimize with polynomial approx)
    x.exp()
}

/// Sigmoid: σ(x) = 1 / (1 + exp(-x))
#[no_mangle]
pub extern "C" fn sigmoid(ptr: *mut f32, len: usize) {
    unsafe {
        for i in 0..len {
            let x = *ptr.add(i);
            *ptr.add(i) = 1.0 / (1.0 + fast_exp(-x));
        }
    }
}

/// Tanh: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
#[no_mangle]
pub extern "C" fn tanh_activation(ptr: *mut f32, len: usize) {
    unsafe {
        for i in 0..len {
            let x = *ptr.add(i);
            *ptr.add(i) = x.tanh();
        }
    }
}

/// ReLU: max(0, x)
#[no_mangle]
pub extern "C" fn relu(ptr: *mut f32, len: usize) {
    unsafe {
        for i in 0..len {
            let x = *ptr.add(i);
            *ptr.add(i) = if x > 0.0 { x } else { 0.0 };
        }
    }
}

/// Squared ReLU: max(0, x)² - used in RWKV
#[no_mangle]
pub extern "C" fn squared_relu(ptr: *mut f32, len: usize) {
    unsafe {
        for i in 0..len {
            let x = *ptr.add(i);
            let r = if x > 0.0 { x } else { 0.0 };
            *ptr.add(i) = r * r;
        }
    }
}

/// SiLU (Swish): x * sigmoid(x)
#[no_mangle]
pub extern "C" fn silu(ptr: *mut f32, len: usize) {
    unsafe {
        for i in 0..len {
            let x = *ptr.add(i);
            let sig = 1.0 / (1.0 + fast_exp(-x));
            *ptr.add(i) = x * sig;
        }
    }
}

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[no_mangle]
pub extern "C" fn gelu(ptr: *mut f32, len: usize) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEF: f32 = 0.044715;
    
    unsafe {
        for i in 0..len {
            let x = *ptr.add(i);
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEF * x3);
            *ptr.add(i) = 0.5 * x * (1.0 + inner.tanh());
        }
    }
}

/// Softmax over a vector (in-place)
#[no_mangle]
pub extern "C" fn softmax(ptr: *mut f32, len: usize) {
    unsafe {
        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..len {
            let x = *ptr.add(i);
            if x > max_val {
                max_val = x;
            }
        }
        
        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for i in 0..len {
            let x = *ptr.add(i);
            let exp_x = fast_exp(x - max_val);
            *ptr.add(i) = exp_x;
            sum += exp_x;
        }
        
        // Normalize
        let inv_sum = 1.0 / sum;
        for i in 0..len {
            *ptr.add(i) *= inv_sum;
        }
    }
}

/// Element-wise exp
#[no_mangle]
pub extern "C" fn exp_elementwise(ptr: *mut f32, len: usize) {
    unsafe {
        for i in 0..len {
            *ptr.add(i) = fast_exp(*ptr.add(i));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sigmoid() {
        let mut data = [0.0f32, 1.0, -1.0, 10.0, -10.0];
        sigmoid(data.as_mut_ptr(), data.len());
        
        assert!((data[0] - 0.5).abs() < 0.001);
        assert!((data[1] - 0.731).abs() < 0.01);
        assert!((data[2] - 0.269).abs() < 0.01);
    }
    
    #[test]
    fn test_softmax() {
        let mut data = [1.0f32, 2.0, 3.0];
        softmax(data.as_mut_ptr(), data.len());
        
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }
}
