// ODIN WASM Kernel: Matrix Multiplication
// =========================================
// Ottimizzato per esecuzione browser via WebAssembly.
// Compile: cargo build --target wasm32-unknown-unknown --release

/// Matrix multiplication kernel
/// C = A @ B
/// A: (M, K), B: (K, N), C: (M, N)
#[no_mangle]
pub extern "C" fn matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        for i in 0..m {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for p in 0..k {
                    let a_val = *a_ptr.add(i * k + p);
                    let b_val = *b_ptr.add(p * n + j);
                    sum += a_val * b_val;
                }
                *c_ptr.add(i * n + j) = sum;
            }
        }
    }
}

/// Tiled matrix multiplication for better cache utilization
/// Uses 4x4 tiles
#[no_mangle]
pub extern "C" fn matmul_tiled(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    const TILE: usize = 4;
    
    unsafe {
        // Initialize C to zero
        for i in 0..(m * n) {
            *c_ptr.add(i) = 0.0;
        }
        
        // Tiled computation
        for i0 in (0..m).step_by(TILE) {
            for j0 in (0..n).step_by(TILE) {
                for p0 in (0..k).step_by(TILE) {
                    // Process tile
                    let i_end = (i0 + TILE).min(m);
                    let j_end = (j0 + TILE).min(n);
                    let p_end = (p0 + TILE).min(k);
                    
                    for i in i0..i_end {
                        for j in j0..j_end {
                            let mut sum = *c_ptr.add(i * n + j);
                            for p in p0..p_end {
                                sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                            }
                            *c_ptr.add(i * n + j) = sum;
                        }
                    }
                }
            }
        }
    }
}

/// Memory allocation helper
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Memory deallocation helper
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matmul_2x2() {
        let a = [1.0f32, 2.0, 3.0, 4.0];  // 2x2
        let b = [5.0f32, 6.0, 7.0, 8.0];  // 2x2
        let mut c = [0.0f32; 4];
        
        matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2);
        
        // Expected: [[19, 22], [43, 50]]
        assert_eq!(c[0], 19.0);
        assert_eq!(c[1], 22.0);
        assert_eq!(c[2], 43.0);
        assert_eq!(c[3], 50.0);
    }
}
