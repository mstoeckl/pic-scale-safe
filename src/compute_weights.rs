/*
 * Copyright (c) Radzivon Bartoshyk, 10/2024. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::filter_weights::{FilterBounds, FilterWeights};
use crate::image_size::FractIndex;
use crate::math::{ConstPI, ConstSqrt2, Jinc};
use crate::sampler::ResamplingFunction;
use num_traits::{AsPrimitive, Float, Signed};
use std::fmt::Debug;
use std::ops::{AddAssign, Div, MulAssign, Neg};

pub(crate) fn generate_weights<T>(
    function: ResamplingFunction,
    in_start: FractIndex,
    in_end: FractIndex,
    in_size: usize,
    out_size: usize,
) -> FilterWeights<T>
where
    T: Copy
        + Neg
        + Signed
        + Float
        + 'static
        + ConstPI
        + MulAssign<T>
        + AddAssign<T>
        + AsPrimitive<f64>
        + AsPrimitive<i64>
        + AsPrimitive<usize>
        + Jinc<T>
        + ConstSqrt2
        + Default
        + AsPrimitive<i32>
        + Div<T, Output = T>
        + Debug,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    assert!(in_start < in_end); // TODO: introduce special degenerate case if in_start = in_end
    assert!(in_end <= FractIndex::from(in_size));

    // Do index calculations in fixed point, so that they behave essentially the same
    // at the left and the right sides of the image.
    let fixed_sub = |from: FractIndex, to: FractIndex| -> FractIndex {
        let (x, overflow) = to.frac.overflowing_sub(from.frac);
        FractIndex {
            int: to.int - from.int - overflow as usize,
            frac: x,
        }
    };
    let chain_add = |x: (usize, u32, u64), y: (usize, u32, u64)| -> (usize, u32, u64) {
        let (z2, z2c) = x.2.carrying_add(y.2, false);
        let (z1, z1c) = x.1.carrying_add(y.1, z2c);
        let (z0, z0c) = x.0.carrying_add(y.0, z1c);
        assert!(!z0c); // overflow not expected in this file
        (z0, z1, z2)
    };
    let chain_div = |value: (usize, u32, usize), steps: usize| -> (usize, u32, u64) {
        let steps_u128 = steps as u128;
        let x1 = ((value.0 as u128) << u32::BITS) | (value.1 as u128);
        let v1 = x1 / steps_u128;
        let r1 = x1 % steps_u128;
        let x2 = r1 << u64::BITS | (value.2 as u128);
        let v2 = x2 / steps_u128;

        let o1 = (
            ((v1 >> u32::BITS) as usize),
            (v1 & u32::MAX as u128) as u32,
            0,
        );
        let o2 = (
            ((v2 >> (u64::BITS + u32::BITS)) as usize),
            ((v2 >> u64::BITS) & u32::MAX as u128) as u32,
            (v2 & u64::MAX as u128) as u64,
        );
        chain_add(o1, o2)
    };
    let chain_div2 = |value: (usize, u32, u64)| -> (usize, u32, u64) {
        (
            value.0 >> 1,
            (value.1 >> 1) | (((value.0 & 0x1) as u32) << (u32::BITS - 1)),
            (value.2 >> 1) | (((value.1 & 0x1) as u64) << (u64::BITS - 1)),
        )
    };
    let chain_round = |value: (usize, u32, u64)| -> (usize, u32) {
        if value.2 >= (1 << (u64::BITS - 1)) {
            let (v1, c) = value.1.carrying_add(1, false);
            (value.0.saturating_add(c as usize), v1)
        } else {
            (value.0, value.1)
        }
    };

    let resampling_filter = function.get_resampling_filter();

    let e32: T = (1u32 << 16).as_() * (1u32 << 16).as_();
    let de32: T = T::one() / e32;
    let fixed_to_float = |x: FractIndex| -> T { x.int.as_() + de32 * x.frac.as_() };

    let in_width = fixed_sub(in_start, in_end);

    let scale = fixed_to_float(in_width) / out_size.as_();
    // Note that this a) rounds down, so multiplying by out_size won't overflow
    // b) has 1/(1<<(u32::BITS + usize::BITS)) error, so adding together `out_size`
    // steps will have < 1/(1<<(u32::BITS)) error at the end
    let in_step = chain_div((in_width.int, in_width.frac, 0), out_size);
    let half_step = chain_div2(in_step);

    let is_resizable_kernel = resampling_filter.is_resizable_kernel;
    let filter_scale_cutoff = match is_resizable_kernel {
        true => scale.max(1f32.as_()),
        false => 1f32.as_(),
    };
    let filter_base_size = resampling_filter.min_kernel_size * 2.;
    let resampling_function = resampling_filter.kernel;
    let window_func = resampling_filter.window;

    let mut bounds: Vec<FilterBounds> = vec![FilterBounds::new(0, 0); out_size];

    let is_area = resampling_filter.is_area_filter && scale < 1.as_();

    if !is_area {
        let base_size: usize = (filter_base_size.as_() * filter_scale_cutoff).round().as_();
        let kernel_size = base_size;
        let filter_radius = (base_size.as_() / 2.as_()).max(0.5.as_());
        let (filter_radius_int, filter_radius_fract): (usize, u32) = (
            filter_radius.floor().as_(),
            AsPrimitive::<usize>::as_(filter_radius.fract() * e32).min(u32::MAX as usize) as u32,
        );
        let filter_scale = 1f32.as_() / filter_scale_cutoff;
        let mut weights: Vec<T> = vec![T::default(); kernel_size * out_size];
        let mut local_filters = vec![T::default(); kernel_size];
        let mut filter_position = 0usize;
        let blur_scale = match window_func {
            None => 1f32.as_(),
            Some(window) => {
                if window.blur.as_() > 0f32.as_() {
                    1f32.as_() / window.blur.as_()
                } else {
                    0f32.as_()
                }
            }
        };

        let mut center_acc = chain_add(half_step, (in_start.int, in_start.frac, 0));
        for bound in bounds.iter_mut() {
            let (center_int, center_frac) = chain_round(center_acc);
            center_acc = chain_add(center_acc, in_step);

            let mut weights_sum: T = 0f32.as_();

            let mut start: usize = center_int.saturating_sub(filter_radius_int);
            if filter_radius_fract > center_frac {
                start = start.saturating_sub(1);
            }
            let (ec, carry) = filter_radius_fract.overflowing_add(center_frac);
            let mut end: usize = center_int.saturating_add(filter_radius_int);
            end = end.saturating_add((ec > 0) as usize + carry as usize);
            end = end.min(in_size).min(start.saturating_add(kernel_size));

            for (local_filter_iteration, k) in (start..end).enumerate() {
                let mut dx_int = k.abs_diff(center_int).as_();
                if k < center_int {
                    dx_int = -dx_int;
                }
                let mut dx_frac = center_frac.abs_diff(0x80000000).as_() * de32;
                if center_frac > 0x80000000 {
                    dx_frac = -dx_frac;
                };

                let dx = dx_int + dx_frac;

                let weight;
                if let Some(resampling_window) = window_func {
                    let mut x = dx.abs();
                    x = if resampling_window.blur.as_() > 0f32.as_() {
                        x * blur_scale
                    } else {
                        x
                    };
                    x = if x <= resampling_window.taper.as_() {
                        0f32.as_()
                    } else {
                        (x - resampling_window.taper.as_())
                            / (1f32.as_() - resampling_window.taper.as_())
                    };
                    let window_producer = resampling_window.window;
                    let x_kernel_scaled = x * filter_scale;
                    let window = if x < resampling_window.window_size.as_() {
                        window_producer(x_kernel_scaled * resampling_window.window_size.as_())
                    } else {
                        0f32.as_()
                    };
                    weight = window * resampling_function(x_kernel_scaled);
                } else {
                    let dx = dx.abs();
                    weight = resampling_function(dx * filter_scale);
                }
                weights_sum += weight;
                local_filters[local_filter_iteration] = weight;
            }

            let size = end - start;

            *bound = FilterBounds::new(start, size);

            if weights_sum != 0f32.as_() {
                let recpeq = 1f32.as_() / weights_sum;

                for (dst, src) in weights
                    .iter_mut()
                    .skip(filter_position)
                    .take(size)
                    .zip(local_filters.iter().take(size))
                {
                    *dst = *src * recpeq;
                }
            } else if size > 0 {
                for dst in weights.iter_mut().skip(filter_position).take(size) {
                    *dst = 0f32.as_();
                }
                let center_tap = center_int - start;
                weights[filter_position + center_tap.min(size - 1)] = 1f32.as_();
            }

            filter_position += kernel_size;
        }

        FilterWeights::<T>::new(
            weights,
            kernel_size,
            kernel_size,
            out_size,
            filter_radius.as_(),
            bounds,
        )
    } else {
        // Simulating INTER_AREA from OpenCV, for up scaling here,
        // this is necessary because weight computation is different
        // from any other func
        let inv_scale: T = 1.as_() / scale;
        let kernel_size = 2;
        let filter_radius: T = 1.as_();
        let mut weights: Vec<T> = vec![T::default(); kernel_size * out_size];
        let mut local_filters = vec![T::default(); kernel_size];
        let mut filter_position = 0usize;

        //
        //       s_k           s_{k+1}          s_{k+2}
        //        |               |               |
        // In *------------------------*------------------------*
        //              j=h                      j=h+1
        // Out    +---------------+---------------+
        //               i=k             i=k+1
        //
        //        w_{k,h} = 1       w_{k+1,h} =
        //                            ratio(ceil(s_{k+1}) - s_{k+1},
        //                              s_{k+2} - floor(s_{k+2}))
        //                          w_{k+1,h+1} = 1 - w_{k+1,h}
        //
        //  where ratio(x,y) = x/(x+y)
        //

        let mut pos_acc = (in_start.int, in_start.frac, 0);

        for bound in bounds.iter_mut() {
            let s_0 = chain_round(pos_acc);
            pos_acc = chain_add(pos_acc, in_step);
            let s_1 = chain_round(pos_acc);

            let mut weights_sum: T = 0f32.as_();

            let (start, end, weight0, weight1) = if s_0.0 == s_1.0 || s_1.1 == 0 {
                (s_0.0, s_0.0 + 1, T::one(), T::zero())
            } else {
                let r = if s_1.0 == s_0.0 + 1 {
                    s_1.1.as_() * de32
                } else {
                    1.as_()
                };
                let w1 = (r * inv_scale).min(1.as_());
                let w0 = 1.as_() - w1;
                (s_0.0, s_0.0 + 2, w0, w1)
            };

            local_filters[0] = weight0;
            local_filters[1] = weight1;

            let size = end - start;

            weights_sum += weight0;
            if size > 1 {
                weights_sum += weight1;
            }
            *bound = FilterBounds::new(start, size);

            if weights_sum != 0f32.as_() {
                let recpeq = 1f32.as_() / weights_sum;

                for (dst, src) in weights
                    .iter_mut()
                    .skip(filter_position)
                    .take(size)
                    .zip(local_filters.iter().take(size))
                {
                    *dst = *src * recpeq;
                }
            } else {
                weights[filter_position] = 1.as_();
            }

            filter_position += kernel_size;
        }

        FilterWeights::new(
            weights,
            kernel_size,
            kernel_size,
            out_size,
            filter_radius.as_(),
            bounds,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_window_has_weights_summing_to_one() {
        let functions = [
            ResamplingFunction::Bilinear,
            ResamplingFunction::Lanczos3,
            ResamplingFunction::Gaussian,
            ResamplingFunction::Bartlett,
            ResamplingFunction::BartlettHann,
            ResamplingFunction::Lanczos3Jinc,
        ];
        for function in functions {
            for (in_size, out_size) in [(100usize, 37usize), (37, 100), (256, 256), (5, 1)] {
                let filter = generate_weights::<f64>(
                    function,
                    FractIndex::from(0),
                    FractIndex::from(in_size),
                    in_size,
                    out_size,
                );
                for (i, chunk) in filter
                    .weights
                    .chunks_exact(filter.kernel_size)
                    .enumerate()
                    .take(out_size)
                {
                    let sum: f64 = chunk.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "{function:?} {in_size}->{out_size}: window {i} sums to {sum}"
                    );
                }
            }
        }
    }
}
